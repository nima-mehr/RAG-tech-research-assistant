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

    def search(
        self,
        query_embedding,
        results: int = 3,
        where: dict | None = None,
        include_embeddings: bool = False,
    ) -> dict:
        vector = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        kwargs = {
            "query_embeddings": [vector],
            "n_results": max(1, results),
            "include": include,
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def delete_by_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def list_sources(self) -> dict[str, int]:
        data = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            name = (meta or {}).get("source")
            if name:
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def count(self) -> int:
        return self.collection.count()
