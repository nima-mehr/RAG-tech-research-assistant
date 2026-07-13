from services.embeddings import EmbeddingModel
from services.vector_store import VectorStore


# The question
question = "Which VPN protocol is faster?"


# Convert question into embedding
embedding_model = EmbeddingModel()

query_vector = embedding_model.create_embeddings(
    [question]
)[0]


# Search database
db = VectorStore()

results = db.search(query_vector, results=3)


print("\nTop results:\n")

for i, document in enumerate(results["documents"][0]):
    print("-" * 50)
    print(f"Result {i+1}:")
    print(document)