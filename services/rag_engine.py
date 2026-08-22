from pathlib import Path

import ollama

from config import CHUNK_OVERLAP, CHUNK_SIZE, OLLAMA_HOST, OLLAMA_MODEL, TOP_K
from services.chunker import chunk_text
from services.embeddings import EmbeddingModel
from services.pdf_loader import load_pdf
from services.vector_store import VectorStore


class RAGEngine:
    def __init__(
        self,
        model: str | None = None,
        ollama_host: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
    ):
        self.model = model or OLLAMA_MODEL
        self.client = ollama.Client(host=ollama_host or OLLAMA_HOST)
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.top_k = top_k or TOP_K
        self.embedding_model = EmbeddingModel()
        self.db = VectorStore()

    def reset(self) -> None:
        self.db.reset()

    def process_pdf(self, pdf_path: str, replace: bool = True) -> int:
        text = load_pdf(pdf_path)
        if not text.strip():
            raise ValueError("No extractable text found in the PDF.")

        chunks = chunk_text(text, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            raise ValueError("PDF produced no usable text chunks.")

        if replace:
            self.db.reset()

        vectors = self.embedding_model.create_embeddings(chunks)
        source = Path(pdf_path).name
        metadatas = [{"source": source, "chunk": index} for index in range(len(chunks))]
        return self.db.add_documents(chunks, vectors, metadatas=metadatas)

    def ask(self, question: str, top_k: int | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if self.db.count() == 0:
            raise ValueError("No documents have been processed yet.")

        k = top_k or self.top_k
        query_embedding = self.embedding_model.create_embeddings([question])[0]
        results = self.db.search(query_embedding, results=k)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        if not documents:
            return {
                "answer": "I don't know based on the document.",
                "sources": [],
            }

        context = "\n\n".join(documents)
        prompt = (
            "Answer the question using ONLY the context below.\n"
            "If the answer is not in the context, say: "
            '"I don\'t know based on the provided document."\n\n'
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        sources = []
        for index, document in enumerate(documents):
            meta = metadatas[index] if index < len(metadatas) else {}
            sources.append(
                {
                    "text": document,
                    "source": meta.get("source"),
                    "chunk": meta.get("chunk"),
                }
            )

        return {
            "answer": response["message"]["content"],
            "sources": sources,
        }
