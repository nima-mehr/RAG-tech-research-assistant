# Tech Research Assistant

Local retrieval-augmented generation (RAG) app for asking questions about PDF documents.

The pipeline extracts text page by page, chunks it, embeds the chunks in batches, stores them in ChromaDB, retrieves a wide candidate set, reranks it with a cross-encoder, and answers with a local Ollama model.

## Features

- Single large books or multiple PDFs in one library
- Page-aware extraction and citations
- Progress feedback while indexing large files
- Batched embeddings (resilient multi-file ingest)
- Clear errors for scanned / image-only PDFs
- Recursive text chunking
- Persistent ChromaDB storage (add / replace / remove by file)
- Candidate over-retrieval + cross-encoder reranking
- Optional MMR so near-duplicate pages do not crowd the context
- Light lexical overlap bonus for exact terms
- Local LLM answers via Ollama
- Streamlit UI for upload, library management, and Q&A
- Settings via environment variables

## Project structure

```
├── app.py                 # Streamlit UI
├── config.py              # Runtime settings
├── services/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── reranker.py
│   ├── retriever.py
│   └── rag_engine.py
├── tests/
│   └── test_pipeline.py
├── pdfs/sample.pdf
├── database/              # ChromaDB data (generated)
├── .env.example
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Install and start Ollama, then pull a model:

```bash
ollama pull llama3
ollama serve
```

Default Ollama URL is `http://127.0.0.1:11434`. Override it in `.env` if needed.

## Run

```bash
streamlit run app.py
```

Upload one or more PDFs, process them, then ask questions. Re-processing the same filename replaces only that file. Check **Replace entire library** to wipe the index first.

Large text PDFs are supported: pages are extracted individually, chunked with page metadata, and embedded in configurable batches with a live progress bar. If one file in a multi-upload fails (e.g. scanned PDF), earlier successful files stay in the library. Scanned image-only PDFs need OCR first (not included yet).

## Tests

```bash
python -m pytest tests/test_pipeline.py
```

These cover PDF loading, chunking, and retrieval scoring without requiring Ollama. The cross-encoder is not downloaded in tests.

## Configuration

See `.env.example` for:

- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE` / `CHUNK_OVERLAP`
- `TOP_K` / `CANDIDATE_K`
- `RERANK_MODEL` / `ENABLE_RERANK`
- `ENABLE_MMR` / `MMR_LAMBDA`
- `EMBED_BATCH_SIZE`
- `CHROMA_PATH` / `COLLECTION_NAME`
- `UPLOAD_DIR`

## Next development steps

- Chat history / multi-turn context
- OCR for scanned books
- Docker / deployment config
