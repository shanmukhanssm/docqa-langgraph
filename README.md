# DOC-QUERY — LangGraph Document Q&A Agent

DOC-QUERY is a command-line study assistant built with [LangGraph](https://langchain-ai.github.io/langgraph/).
Give it a **PDF, DOCX or TXT** document and it will (1) validate the file,
(2) extract the text page by page, (3) summarize every page into short bullet
points using an LLM, and then (4) answer your questions by first locating the
single most relevant page and reading that full page.

The whole agent is a 4-node LangGraph `StateGraph` with conditional routing,
and every LLM conversation goes through **one tiny function** (`ask_llm` in
`llm.py`) to keep the project easy to read and extend.

## How it works

```
                 ┌──────────────────────────────────────────────┐
  new file  ───► │ NODE 1 preprocess ──► NODE 2 extract ────►  │
                 │   (pdf/docx/txt?)      (page-wise text)      │
                 │        │                   │                 │
                 │      reject              accept              │
                 │        ▼                   ▼                 │
                 │       END            NODE 3 summarize ──► END │
                 │                       (parallel LLM calls,   │
                 │                        summaries.csv)        │
                 ├──────────────────────────────────────────────┤
  question  ───► │ NODE 4 answer ──────────────────────────► END│
                 │  (pick best page from summaries,             │
                 │   then answer from that full page)           │
                 └──────────────────────────────────────────────┘
```

| Node | File | What it does |
|------|------|--------------|
| 1. preprocess | `nodes.py` | Rejects missing files and unsupported formats (only `.pdf`, `.docx`, `.txt`). |
| 2. extract | `nodes.py` | Reads the file page by page (`pypdf` / `python-docx` / plain text) and saves `extracted_text.txt` with `--- PAGE n ---` markers. |
| 3. summarize | `nodes.py` | Splits every page into ~3,000-character chunks, summarizes all chunks **in parallel** (ThreadPoolExecutor), merges multi-chunk pages, and writes `summaries.csv` (`page_code`, `summary_point`). |
| 4. answer | `nodes.py` | Asks the LLM which single page (from all summary points) most likely holds the answer, then answers strictly from that page's full text. Questions unrelated to the document are politely rejected. |

`graph.py` wires the nodes together with conditional edges so a fresh file
starts at node 1 while a question jumps straight to node 4.

## Project structure

```
docqa-langgraph/
├── main.py          # CLI entry point: file input + question loop
├── graph.py         # LangGraph StateGraph definition & routing
├── nodes.py         # the 4 nodes + chunking/summarization helpers
├── llm.py           # the single LLM gateway (OpenAI-compatible client)
├── requirements.txt # dependencies
└── test_output.txt  # console transcript of the successful end-to-end test
```

## Setup

```bash
pip install -r requirements.txt
```

The LLM is accessed through any **OpenAI-compatible API**. By default the
project targets **Groq** (`llama-3.3-70b-versatile`), so all you need is a
free Groq key:

```bash
export API_KEY="your_groq_key"          # windows: set API_KEY=your_groq_key
python main.py
```

All three connection settings can be overridden with environment variables,
so you can point the agent at any OpenAI-compatible endpoint (Groq, OpenAI,
a local server, ...):

| Variable      | Default                          | Purpose                     |
|---------------|----------------------------------|-----------------------------|
| `API_KEY`     | – (required)                     | API key for the LLM service |
| `MODEL_NAME`  | `llama-3.3-70b-versatile`        | Model id                    |
| `BASE_URL`    | `https://api.groq.com/openai/v1` | OpenAI-compatible base URL  |

## Usage

```text
$ python main.py
==============================================================
          DOC-QUERY  (LangGraph Document Q&A Agent)
==============================================================

Enter path of your file (PDF / DOCX / TXT): my_document.pdf
[NODE 1] File is a valid .pdf file. Accepted.
[NODE 2] Extracted 30 page(s). Full text saved to extracted_text.txt
[NODE 3] Summarizing the document with LLM (parallel calls)...

Document ready! (30 page(s) processed)
Now you can ask questions about the document.

Your question: What is the largest planet in the Solar System?
ANSWER (found on page 5):
Jupiter is the largest planet in the Solar System.
```

Artifacts written next to the program: `extracted_text.txt` (full page-wise
text) and `summaries.csv` (one bullet point per row with its `page_code`).

## Testing

An end-to-end test was run against a 30-page PDF compiled from **four
Wikipedia articles** (Solar System, Photosynthesis, Internet, Mount Everest —
roughly 7-8 pages each). The transcript of the successful run is in
[`test_output.txt`](test_output.txt). The test asked one question per topic
plus one deliberately off-topic question:

| Question | Expected | Result |
|----------|----------|--------|
| Largest planet in the Solar System? | Jupiter, page 1-8 | page 5 — "Jupiter is the largest planet in the Solar System." |
| Main pigment used in photosynthesis? | Chlorophyll, page 9-15 | page 13 — "Chlorophyll." |
| Precursor network of the Internet? | ARPANET, page 16-22 | page 18 — "ARPANET." |
| Height of Mount Everest? | ~8,848 m, page 23-30 | page 28 — "8,848.86 metres (29,031.7 ft)" |
| Who wrote the play Hamlet? (off-topic) | polite rejection | "This question does not seem to be related to this document." |

The run used the project's `BASE_URL` override with a local
OpenAI-compatible endpoint, which is exactly how the agent can be pointed at
Groq or any other provider in real use.

## Requirements

- Python 3.10+
- `langgraph`, `openai`, `pypdf`, `python-docx` (see `requirements.txt`)
- An API key from [groq.com](https://console.groq.com/keys) (free) or any other OpenAI-compatible service
