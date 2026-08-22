# Tech Research Assistant

Local retrieval-augmented generation (RAG) app for asking questions about PDF documents.

The pipeline extracts text, chunks it, embeds the chunks, stores them in ChromaDB, retrieves relevant context, and answers with a local Ollama model.

## Features

- PDF text extraction
- Recursive text chunking
- Sentence Transformer embeddings
- Persistent ChromaDB storage
- Local LLM answers via Ollama
- Streamlit UI for upload, processing, and Q&A
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

Upload a PDF, process it, then ask questions. Answers are restricted to retrieved document context.

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

## Next development steps

- Multi-document collections without replacing the current index
- Chat history / multi-turn context
- Page-level citations
- Retrieval reranking
- Docker / deployment config
