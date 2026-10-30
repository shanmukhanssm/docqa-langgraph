
import os
import sys

from graph import build_graph
from llm import API_KEY


def main():
    print("=" * 62)
    print("          DOC-QUERY  (LangGraph Document Q&A Agent)")
    print("=" * 62)

    # basic check before starting - make sure we have an API key
    # (the key is hardcoded in llm.py; the env var API_KEY can override it)
    if not API_KEY or API_KEY == "not_set_yet":
        print("\n[ERROR] No API key configured!")
        print("Set your key in llm.py (API_KEY) or via the environment:")
        print('   export API_KEY="your_key"        (windows: $env:API_KEY="your_key")')
        sys.exit(1)

    app = build_graph()

    # ------------------------------------------------------------
    # PART A : take the file from user and run nodes 1 -> 2 -> 3
    # ------------------------------------------------------------
    file_path = input("\nEnter path of your file (PDF / DOCX / TXT): ").strip().strip('"').strip("'")

    if not file_path:
        print("No file path entered. Exiting.")
        sys.exit(1)

    print("\nProcessing your document (nodes 1 -> 3), please wait...")
    state = app.invoke({"file_path": file_path})

    # if node 1 rejected the file or node 2 found no text, we stop here
    if not state.get("file_ok") or state.get("num_pages", 0) == 0:
        print("\n[REJECTED]", state.get("error_msg", "Invalid file."))
        sys.exit(1)

    print(f"\nDocument ready! ({state.get('num_pages')} page(s) processed)")
    print("Page-wise summary points stored in summaries.csv")
    print("\nNow you can ask questions about the document.")
    print("Type 'exit' or 'quit' to stop.\n")

    # ------------------------------------------------------------
    # PART B : question-answer loop (node 4 runs for every question)
    # ------------------------------------------------------------
    while True:
        try:
            question = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye bye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q", "bye"):
            print("Bye bye!")
            break

        result = app.invoke({"question": question})

        print("\n" + "-" * 62)
        print(f"ANSWER (found on page {result.get('best_page', '?')}):")
        print("-" * 62)
        print(result.get("answer", "Sorry, no answer was generated."))
        print()


if __name__ == "__main__":
    main()
