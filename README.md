# Tech Research Assistant

Local retrieval-augmented generation (RAG) app for asking questions about PDF documents.

The pipeline extracts text page by page, chunks it, embeds the chunks, stores them in ChromaDB, retrieves relevant context, and answers with a local Ollama model.

## Features

- Single large books or multiple PDFs in one library
- Page-aware extraction and citations
- Recursive text chunking
- Batched Sentence Transformer embeddings
- Persistent ChromaDB storage (add / replace / remove by file)
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

A 400-page text PDF is supported: pages are extracted individually, chunked with page metadata, and embedded in batches. Expect a few minutes on CPU depending on length. Scanned image-only PDFs need OCR (not included yet).

## Tests

```bash
python -m pytest tests/test_pipeline.py
```

These cover PDF loading and chunking without requiring Ollama.

## Configuration

See `.env.example` for:

- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE` / `CHUNK_OVERLAP`
- `TOP_K`
- `CHROMA_PATH` / `COLLECTION_NAME`
- `UPLOAD_DIR`

## Next development steps

- OCR for scanned books
- Chat history / multi-turn context
- Retrieval reranking
- Background / progress for very large ingest
- Docker / deployment config
