import chromadb
from chromadb.config import Settings


class VectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="database"
        )

        self.collection = self.client.get_or_create_collection(
            name="documents"
        )


    def add_documents(self, chunks, embeddings):

        ids = [
            str(i) for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )


    def search(self, query_embedding, results=3):

        result = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=results
        )

        return result