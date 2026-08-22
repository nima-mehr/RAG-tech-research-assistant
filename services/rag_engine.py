from pathlib import Path

import ollama

from config import CHUNK_OVERLAP, CHUNK_SIZE, OLLAMA_HOST, OLLAMA_MODEL, TOP_K
from services.chunker import chunk_pages
from services.embeddings import EmbeddingModel
from services.pdf_loader import load_pdf_pages
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

    def list_documents(self) -> dict[str, int]:
        return self.db.list_sources()

    def remove_document(self, source: str) -> None:
        self.db.delete_by_source(source)

    def process_pdf(
        self,
        pdf_path: str,
        replace: bool = False,
        replace_source: bool = True,
    ) -> int:
        pages = load_pdf_pages(pdf_path)
        if not pages:
            raise ValueError("No extractable text found in the PDF.")

        records = chunk_pages(
            pages,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        if not records:
            raise ValueError("PDF produced no usable text chunks.")

        if replace:
            self.db.reset()

        source = Path(pdf_path).name
        if replace_source and not replace:
            self.db.delete_by_source(source)

        texts = [record["text"] for record in records]
        vectors = self.embedding_model.create_embeddings(texts)
        metadatas = [
            {
                "source": record["source"],
                "page": int(record["page"]),
                "chunk": int(record["chunk"]),
            }
            for record in records
        ]
        return self.db.add_documents(texts, vectors, metadatas=metadatas)

    def process_pdfs(self, pdf_paths: list[str], replace: bool = False) -> int:
        if replace:
            self.db.reset()

        total = 0
        for path in pdf_paths:
            total += self.process_pdf(path, replace=False, replace_source=True)
        return total

    def ask(self, question: str, top_k: int | None = None, source: str | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if self.db.count() == 0:
            raise ValueError("No documents have been processed yet.")

        k = top_k or self.top_k
        # Slightly raise k for larger libraries so one book does not starve others.
        library_size = self.db.count()
        if top_k is None and library_size > 400:
            k = max(k, 6)
        k = min(k, library_size)

        query_embedding = self.embedding_model.create_embeddings([question])[0]
        where = {"source": source} if source else None
        results = self.db.search(query_embedding, results=k, where=where)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        if not documents:
            return {
                "answer": "I don't know based on the provided documents.",
                "sources": [],
            }

        context_blocks = []
        sources = []
        for index, document in enumerate(documents):
            meta = metadatas[index] if index < len(metadatas) else {}
            label = meta.get("source") or "document"
            page = meta.get("page")
            heading = f"[{label}"
            if page is not None:
                heading += f", p. {page}"
            heading += "]"
            context_blocks.append(f"{heading}\n{document}")
            sources.append(
                {
                    "text": document,
                    "source": label,
                    "page": page,
                    "chunk": meta.get("chunk"),
                }
            )

        context = "\n\n".join(context_blocks)
        prompt = (
            "Answer the question using ONLY the context below.\n"
            "Cite the source filename and page number when you use a passage.\n"
            "If the answer is not in the context, say: "
            '"I don\'t know based on the provided documents."\n\n'
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "answer": response["message"]["content"],
            "sources": sources,
        }
