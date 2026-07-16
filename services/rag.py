import ollama

from services.embeddings import EmbeddingModel
from services.vector_store import VectorStore


class RAG:

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.db = VectorStore()

    def ask(self, question):

        # Create embedding for the question
        query_embedding = self.embedding_model.create_embeddings(
            [question]
        )[0]

        # Retrieve relevant chunks
        results = self.db.search(
            query_embedding,
            results=3
        )

        context = "\n\n".join(
            results["documents"][0]
        )

        prompt = f"""
Answer the question using ONLY the context below.

If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

        client = ollama.Client(
        host="http://127.0.0.1:12000"
        )

        response = client.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]