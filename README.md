# Tech Research Assistant - RAG Application

A local Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents and receive answers generated from the document content.

The project combines document processing, semantic search, vector databases, and local Large Language Models (LLMs) to build an AI-powered research assistant.

## Features

* PDF document loading and text extraction
* Intelligent text chunking
* Text embeddings using Sentence Transformers
* Vector storage and similarity search with ChromaDB
* Local LLM response generation using Ollama
* Context-aware answers based only on uploaded documents
* Fully local operation (no external AI API required)

## Architecture

```
PDF Document
      |
      v
Text Extraction
      |
      v
Text Chunking
      |
      v
Embedding Generation
(Sentence Transformers)
      |
      v
Vector Database
(ChromaDB)
      |
      v
User Question
      |
      v
Query Embedding
      |
      v
Similarity Search
      |
      v
Retrieved Context
      |
      v
Local LLM
(Ollama)
      |
      v
Generated Answer
```

## Technologies Used

### Backend

* Python
* PyPDF
* Sentence Transformers
* ChromaDB
* Ollama
* LangChain Text Splitters

### AI Models

* Embedding model:

  * `sentence-transformers/all-MiniLM-L6-v2`

* Language model:

  * Ollama compatible local LLMs (for example Llama 3)

## Project Structure

```
tech-research-assistant/

├── services/
│   ├── pdf_loader.py        # PDF text extraction
│   ├── chunker.py           # Document chunking
│   ├── embeddings.py        # Embedding generation
│   ├── vector_store.py      # ChromaDB interface
│   └── rag.py               # Complete RAG pipeline
│
├── pdfs/
│   └── sample.pdf           # Example document
│
├── database/                # ChromaDB storage
│
├── test_pdf.py              # PDF extraction test
├── test_embeddings.py       # Embedding generation test
├── test_vector_store.py     # Vector database test
├── test_search.py           # Retrieval test
├── test_rag.py              # Full RAG pipeline test
│
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>

cd tech-research-assistant
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv

.\venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv

source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama:

https://ollama.com

Download a local model:

```bash
ollama pull llama3
```

Start the Ollama server:

```bash
ollama serve
```

The default Ollama address is:

```
http://127.0.0.1:11434
```

If the default port is unavailable, Ollama can be configured with another port:

Windows PowerShell:

```powershell
$env:OLLAMA_HOST="127.0.0.1:12000"

ollama serve
```

## Usage

### 1. Add PDF documents

Place PDF files inside:

```
pdfs/
```

### 2. Create embeddings and store documents

Run:

```bash
python test_vector_store.py
```

This will:

* Extract PDF text
* Split the document into chunks
* Generate embeddings
* Store vectors in ChromaDB

### 3. Test document retrieval

Run:

```bash
python test_search.py
```

Example:

```
Question:
Which VPN protocol is faster?

Retrieved context:
WireGuard is generally faster than OpenVPN.
```

### 4. Ask questions using RAG

Run:

```bash
python test_rag.py
```

Example:

```
Question:
Which VPN protocol is faster?

Answer:
WireGuard is generally faster than OpenVPN.
```

## How RAG Works in This Project

Retrieval-Augmented Generation combines search and generation:

1. The user's question is converted into an embedding vector.
2. ChromaDB searches for the most relevant document chunks.
3. Retrieved information is added to the LLM prompt.
4. The LLM generates an answer based on the provided context.

This reduces hallucination because the model is instructed to answer only from retrieved documents.

## Future Improvements

* Web interface using Streamlit
* PDF upload system
* Chat history and multi-turn conversations
* Source citations with page numbers
* Multiple document support
* Improved retrieval with reranking
* Authentication and user management
* Cloud deployment

## License

This project is currently for educational and research purposes.
