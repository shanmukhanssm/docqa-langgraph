
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from pypdf import PdfReader
from docx import Document

from llm import ask_llm

ALLOWED_FORMATS = [".pdf", ".docx", ".txt"]

EXTRACTED_TEXT_FILE = "extracted_text.txt"   
SUMMARY_CSV_FILE = "summaries.csv"           
CHUNK_SIZE = 3000   
MAX_WORKERS = 5     


class GraphState(TypedDict):
    """this state dictionary is passed around between all the nodes"""
    file_path: str
    file_ok: bool
    error_msg: str
    pages: list            
    num_pages: int
    question: str
    best_page: int
    answer: str


def preprocess_node(state: GraphState) -> dict:
    """Checks the file path and simply rejects the file if the
    format is not pdf / docx / txt."""
    print("\n[NODE 1] Preprocessing the input file...")

    file_path = state.get("file_path", "")


    if not file_path or not os.path.exists(file_path):
        print(f"[NODE 1] File '{file_path}' not found. Rejecting.")
        return {"file_ok": False,
                "error_msg": "File not found. Please check the path and try again."}

    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ALLOWED_FORMATS:
        print(f"[NODE 1] '{ext}' format is NOT supported. Rejecting.")
        return {"file_ok": False,
                "error_msg": f"Unsupported format '{ext}'. Only PDF, DOCX and TXT files are allowed."}

    print(f"[NODE 1] File is a valid {ext} file. Accepted.")
    return {"file_ok": True, "error_msg": ""}


# ==================================================================
# NODE 2 : PROCESSING  (text extraction)
# ==================================================================
def extract_node(state: GraphState) -> dict:
    """Extracts the text from the accepted file and saves the full
    text into extracted_text.txt (with page markers)."""
    print("\n[NODE 2] Extracting text from the file...")

    file_path = state["file_path"]
    ext = os.path.splitext(file_path)[1].lower()
    pages = []

    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text.strip())

        elif ext == ".docx":
            doc = Document(file_path)
            full_text = "\n".join(p.text for p in doc.paragraphs)
            pages = [full_text.strip()]

        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                pages = [f.read().strip()]

    except Exception as e:
        print(f"[NODE 2] Failed to read the file -> {e}")
        return {"pages": [], "num_pages": 0,
                "error_msg": "Could not read the file (maybe it is corrupted)."}

    if sum(len(p) for p in pages) == 0:
        print("[NODE 2] No text found in the file.")
        return {"pages": [], "num_pages": 0,
                "error_msg": "No text found in this file (scanned PDFs are not supported)."}

    with open(EXTRACTED_TEXT_FILE, "w", encoding="utf-8") as f:
        for i, page in enumerate(pages):
            f.write(f"--- PAGE {i + 1} ---\n{page}\n\n")

    print(f"[NODE 2] Extracted {len(pages)} page(s). Full text saved to {EXTRACTED_TEXT_FILE}")
    return {"pages": pages, "num_pages": len(pages), "error_msg": ""}



def make_chunks(text: str) -> list:
    """splits a big page text into small chunks of CHUNK_SIZE characters"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE
    return chunks


def summarize_one_chunk(chunk_text: str) -> str:
    """sends ONE chunk to the LLM and gets back a few bullet points"""
    prompt = (
        "Summarize the following text into maximum 4 short bullet points.\n"
        "Reply ONLY with the bullet points, one per line, each starting with '- '.\n\n"
        f"TEXT:\n{chunk_text}"
    )
    try:
        return ask_llm(prompt)
    except Exception as e:
        print(f"[NODE 3] LLM call failed for one chunk -> {e}")
        return ""


def merge_chunk_summaries(page_num: int, points_list: list) -> str:
    """if one page was split into many chunks, we merge all their points
    into 3-4 final bullet points for that page"""
    combined = "\n".join(points_list)
    prompt = (
        f"Below are summary points collected from different parts of page {page_num}.\n"
        "Combine them into exactly 3 to 4 short bullet points for the whole page.\n"
        "Reply ONLY with the bullet points, one per line, each starting with '- '.\n\n"
        f"POINTS:\n{combined}"
    )
    try:
        return ask_llm(prompt)
    except Exception:
        return "\n".join([l for l in combined.splitlines() if l.strip()][:4])


def summarize_node(state: GraphState) -> dict:
    """Chunks every page, sends all chunks to the LLM in parallel,
    and saves page_code + summary points into summaries.csv.
    Every summary point gets the same page_code as its page."""
    print("\n[NODE 3] Summarizing the document with LLM (parallel calls)...")

    pages = state["pages"]

    all_chunks = []         
    for i, page_text in enumerate(pages):
        for chunk in make_chunks(page_text):
            all_chunks.append((i + 1, chunk))

    print(f"[NODE 3] Total {len(all_chunks)} chunk(s) prepared. Sending to LLM in parallel...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunk_summaries = list(executor.map(summarize_one_chunk, [c for (_p, c) in all_chunks]))

    points_by_page = {i + 1: [] for i in range(len(pages))}
    for (page_num, _chunk), summary in zip(all_chunks, chunk_summaries):
        if summary and summary.strip():
            points_by_page[page_num].append(summary.strip())

    final_summaries = {}
    for page_num, points in points_by_page.items():
        if len(points) == 0:
            final_summaries[page_num] = "- (no summary generated for this page)"
        elif len(points) == 1:
            final_summaries[page_num] = points[0]
        else:
            print(f"[NODE 3] Merging chunk summaries of page {page_num}...")
            final_summaries[page_num] = merge_chunk_summaries(page_num, points)

    with open(SUMMARY_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_code", "summary_point"])
        for page_num in sorted(final_summaries):
            for line in final_summaries[page_num].splitlines():
                line = line.strip()
                if line:
                    writer.writerow([page_num, line])

    print(f"[NODE 3] Done! Page-wise summary points saved to {SUMMARY_CSV_FILE}")
    return {}


def load_pages_from_txt() -> list:
    """reads back extracted_text.txt and splits it into pages again"""
    with open(EXTRACTED_TEXT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    pages = []
    for part in content.split("--- PAGE ")[1:]:
        if "---" in part:
            pages.append(part.split("---\n", 1)[1].strip())
        else:
            pages.append(part.strip())
    return pages


def answer_node(state: GraphState) -> dict:
    """Reads summaries.csv -> asks the LLM which page most likely has the
    answer -> loads that FULL page -> asks the LLM to answer from it."""
    print("\n[NODE 4] Looking for the answer...")

    question = state["question"]

    try:
        summary_text = ""
        with open(SUMMARY_CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip the header row
            for row in reader:
                if len(row) >= 2:
                    summary_text += f"Page {row[0]}: {row[1]}\n"
    except FileNotFoundError:
        print("[NODE 4] summaries.csv not found!")
        return {"answer": "No document has been processed yet. Please upload a file first."}

    pick_prompt = (
        "You are given the summary points of every page of a document.\n"
        "Based on the question, decide which ONE page most likely contains the answer.\n"
        "Reply with ONLY the page number (for example: 3).\n"
        "If no page is related to the question at all, reply with only 0.\n\n"
        f"SUMMARY POINTS:\n{summary_text}\n"
        f"QUESTION: {question}"
    )
    try:
        reply = ask_llm(pick_prompt).strip()
        digits = "".join(ch for ch in reply if ch.isdigit())
        page_num = int(digits) if digits else 0
    except Exception as e:
        print(f"[NODE 4] Could not figure out the best page -> {e}")
        return {"answer": "Sorry, I could not figure out which page has the answer."}

    if page_num == 0:
        print("[NODE 4] No page matched the question.")
        return {"answer": "This question does not seem to be related to this document."}

    try:
        pages = load_pages_from_txt()
    except FileNotFoundError:
        return {"answer": "No document has been processed yet. Please upload a file first."}

    if page_num < 1 or page_num > len(pages):
        print(f"[NODE 4] LLM said page {page_num} but the document has only {len(pages)} page(s).")
        return {"answer": "Sorry, I could not find a matching page for your question in this document."}

    print(f"[NODE 4] Page {page_num} looks most relevant. Reading the full page...")

    answer_prompt = (
        "You are given the full content of one page of a document.\n"
        "Answer the user's question using ONLY this page.\n"
        "If the answer is not present in this page, clearly say that you could not find it.\n\n"
        f"PAGE {page_num} CONTENT:\n{pages[page_num - 1]}\n\n"
        f"QUESTION: {question}"
    )
    try:
        answer = ask_llm(answer_prompt)
    except Exception as e:
        print(f"[NODE 4] LLM call failed -> {e}")
        return {"answer": "Sorry, something went wrong while generating the answer."}

    print("[NODE 4] Answer ready.")
    return {"answer": answer, "best_page": page_num}
