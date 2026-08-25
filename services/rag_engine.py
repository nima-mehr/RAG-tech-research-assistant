from collections.abc import Callable
from pathlib import Path

import ollama

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TOP_K,
)
from services.chunker import chunk_pages
from services.embeddings import EmbeddingModel
from services.pdf_loader import load_pdf_document
from services.vector_store import VectorStore

ProgressCallback = Callable[[dict], None]


class RAGEngine:
    def __init__(
        self,
        model: str | None = None,
        ollama_host: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        embed_batch_size: int | None = None,
    ):
        self.model = model or OLLAMA_MODEL
        self.client = ollama.Client(host=ollama_host or OLLAMA_HOST)
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
        self.top_k = top_k or TOP_K
        self.embed_batch_size = embed_batch_size or EMBED_BATCH_SIZE
        self.embedding_model = EmbeddingModel()
        self.db = VectorStore()

    def reset(self) -> None:
        self.db.reset()

    def list_documents(self) -> dict[str, int]:
        return self.db.list_sources()

    def remove_document(self, source: str) -> None:
        self.db.delete_by_source(source)

    @staticmethod
    def _emit(on_progress: ProgressCallback | None, **payload) -> None:
        if on_progress is not None:
            on_progress(payload)

    def process_pdf(
        self,
        pdf_path: str,
        replace: bool = False,
        replace_source: bool = True,
        on_progress: ProgressCallback | None = None,
        file_index: int = 1,
        file_count: int = 1,
    ) -> int:
        source = Path(pdf_path).name
        file_base = (file_index - 1) / max(file_count, 1)
        file_span = 1 / max(file_count, 1)

        self._emit(
            on_progress,
            stage="extract",
            file=source,
            file_index=file_index,
            file_count=file_count,
            fraction=file_base,
            message=f"[{file_index}/{file_count}] Reading {source}…",
        )

        document = load_pdf_document(pdf_path)
        pages = document["pages"]
        total_pages = document["total_pages"]
        text_pages = document["text_pages"]

        if total_pages == 0:
            raise ValueError(f"'{source}' has no pages.")

        if not pages:
            raise ValueError(
                f"No extractable text in '{source}' "
                f"({total_pages} page{'s' if total_pages != 1 else ''}). "
                "This is often a scanned or image-only PDF; OCR is not enabled yet. "
                "Try a text-based PDF, or run OCR on the file first."
            )

        self._emit(
            on_progress,
            stage="chunk",
            file=source,
            file_index=file_index,
            file_count=file_count,
            pages=text_pages,
            total_pages=total_pages,
            fraction=file_base + 0.05 * file_span,
            message=(
                f"[{file_index}/{file_count}] Chunking {source} "
                f"({text_pages}/{total_pages} pages with text)…"
            ),
        )

        records = chunk_pages(
            pages,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        if not records:
            raise ValueError(
                f"'{source}' produced no usable text chunks after splitting. "
                "The file may only contain very short fragments."
            )

        if replace:
            self.db.reset()

        if replace_source and not replace:
            self.db.delete_by_source(source)

        total_chunks = len(records)
        batch_size = max(1, self.embed_batch_size)
        indexed = 0

        for start in range(0, total_chunks, batch_size):
            batch = records[start : start + batch_size]
            texts = [record["text"] for record in batch]
            vectors = self.embedding_model.create_embeddings(texts, batch_size=batch_size)
            metadatas = [
                {
                    "source": record["source"],
                    "page": int(record["page"]),
                    "chunk": int(record["chunk"]),
                }
                for record in batch
            ]
            indexed += self.db.add_documents(texts, vectors, metadatas=metadatas)

            done = min(start + batch_size, total_chunks)
            # Reserve ~10% of this file's span for extract/chunk; rest for embed.
            embed_fraction = done / total_chunks
            fraction = file_base + (0.1 + 0.9 * embed_fraction) * file_span
            self._emit(
                on_progress,
                stage="embed",
                file=source,
                file_index=file_index,
                file_count=file_count,
                chunk=done,
                chunks=total_chunks,
                pages=text_pages,
                total_pages=total_pages,
                fraction=min(1.0, fraction),
                message=(
                    f"[{file_index}/{file_count}] Embedding {source}: "
                    f"{done}/{total_chunks} chunks"
                ),
            )

        self._emit(
            on_progress,
            stage="done",
            file=source,
            file_index=file_index,
            file_count=file_count,
            chunk=indexed,
            chunks=total_chunks,
            pages=text_pages,
            total_pages=total_pages,
            fraction=file_base + file_span,
            message=f"[{file_index}/{file_count}] Finished {source} ({indexed} chunks)",
        )
        return indexed

    def process_pdfs(
        self,
        pdf_paths: list[str],
        replace: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Index multiple PDFs sequentially.

        Returns:
            {
                "chunks": int,
                "files_ok": list[str],
                "errors": list[{"file": str, "error": str}],
            }

        Files that succeed stay in the library even if a later file fails.
        """
        if replace:
            self.db.reset()

        paths = list(pdf_paths)
        file_count = len(paths)
        total_chunks = 0
        files_ok: list[str] = []
        errors: list[dict] = []

        for index, path in enumerate(paths, start=1):
            name = Path(path).name
            try:
                added = self.process_pdf(
                    path,
                    replace=False,
                    replace_source=True,
                    on_progress=on_progress,
                    file_index=index,
                    file_count=file_count,
                )
            except Exception as exc:
                errors.append({"file": name, "error": str(exc)})
                self._emit(
                    on_progress,
                    stage="error",
                    file=name,
                    file_index=index,
                    file_count=file_count,
                    fraction=index / max(file_count, 1),
                    message=f"[{index}/{file_count}] Failed {name}: {exc}",
                )
                continue

            total_chunks += added
            files_ok.append(name)

        self._emit(
            on_progress,
            stage="complete",
            file_count=file_count,
            chunk=total_chunks,
            fraction=1.0 if not errors or files_ok else max(0.0, len(files_ok) / max(file_count, 1)),
            message=(
                f"Done: {len(files_ok)}/{file_count} file(s), {total_chunks} chunks"
                + (f", {len(errors)} failed" if errors else "")
            ),
        )
        return {
            "chunks": total_chunks,
            "files_ok": files_ok,
            "errors": errors,
        }

    def ask(self, question: str, top_k: int | None = None, source: str | None = None) -> dict:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        if self.db.count() == 0:
            raise ValueError("No documents have been processed yet.")

        k = top_k or self.top_k
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
