# PDF Question Answering with Page Citations (RAG)

Ask questions over 3 public PDFs (Union Budget 2026-27 speech, RBI Annual Report 2025-26, RBI Financial Stability Report 2026). Every answer cites the source file and page.

## How it works
1. `ingest.py` reads PDFs with pypdf and splits pages into ~800-character chunks, keeping file name and page number.
2. `build_index.py` embeds chunks with sentence-transformers (all-MiniLM-L6-v2) and stores them in ChromaDB.
3. `ask.py` retrieves the top 4 chunks and asks an LLM (Groq, openai/gpt-oss-120b) to answer only from them, citing sources.
4. `api.py` exposes it as a FastAPI endpoint: `POST /ask`.

## Run it
Install: pip install pypdf chromadb sentence-transformers groq fastapi uvicorn

Put your PDFs in a `data/` folder, then run:

    python build_index.py
    $env:GROQ_API_KEY="your_key"
    uvicorn api:app

Open http://127.0.0.1:8000/docs and try `POST /ask`.

## Evaluation
20 hand-written questions, each with the PDF and page that contains the answer (`eval.py`).
- hit@4: 15/20 = 75%
- hit@10: 16/20 = 80%

## Limitations
- Small test set (20 questions); exact-page match is a strict test.
- Missed questions: real GDP growth projection, RBI surplus transfer, CET1 ratio, 41% devolution share (the answer chunk was ranked outside the top 4).
- Fixed-size chunking can split tables and sentences; scanned PDFs are not supported.
- No re-ranking yet.