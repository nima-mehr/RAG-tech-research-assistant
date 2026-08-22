import uuid

import chromadb

from config import CHROMA_PATH, COLLECTION_NAME


class VectorStore:
    def __init__(self, path: str | None = None, collection_name: str | None = None):
        self.client = chromadb.PersistentClient(path=path or CHROMA_PATH)
        self.collection_name = collection_name or COLLECTION_NAME
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_documents(self, chunks: list[str], embeddings, metadatas: list[dict] | None = None) -> int:
        if not chunks:
            return 0

        ids = [str(uuid.uuid4()) for _ in chunks]
        payload = {
            "ids": ids,
            "documents": chunks,
            "embeddings": embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
        }
        if metadatas:
            payload["metadatas"] = metadatas

        self.collection.add(**payload)
        return len(chunks)

    def search(self, query_embedding, results: int = 3) -> dict:
        vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        return self.collection.query(
            query_embeddings=[vector],
            n_results=results,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()
