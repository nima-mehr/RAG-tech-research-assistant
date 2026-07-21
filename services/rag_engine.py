from streamlit import text

from services.pdf_loader import load_pdf
from services.chunker import chunk_text
from services.embeddings import EmbeddingModel
from services.vector_store import VectorStore

import ollama


class RAGEngine:

    def __init__(
        self,
        model="llama3",
        ollama_host="http://127.0.0.1:12000",
    ):
        self.model = model
        self.client = ollama.Client(host=ollama_host)

        self.embedding_model = EmbeddingModel()
        self.db = VectorStore()
    
    def process_pdf(
        self,
        pdf_path,
        chunk_size=200, 
        overlap=40,
    ):
        text = load_pdf(pdf_path)

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        vectors = self.embedding_model.create_embeddings(chunks)

        self.db.add_documents(
          chunks,
            vectors,
        )

        return len(chunks)
    
    def ask(self, question, top_k=3):

        query_embedding = self.embedding_model.create_embeddings(
            [question]
        )[0]

        results = self.db.search(
            query_embedding,
            results=top_k,
        )

        documents = results["documents"][0]

        context = "\n\n".join(documents)

        prompt = f"""
    Answer ONLY using the context below.

    If the answer is not present,
    say "I don't know based on the document."

    Context:

    {context}

    Question:
    {question}

    Answer:
    """

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return {
            "answer": response["message"]["content"],
            "sources": documents,
        }