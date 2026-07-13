from services.pdf_loader import load_pdf
from services.chunker import chunk_text
from services.embeddings import EmbeddingModel


text = load_pdf("pdfs/sample.pdf")

chunks = chunk_text(
    text,
    chunk_size=50,
    overlap=10
)


embedding_model = EmbeddingModel()

vectors = embedding_model.create_embeddings(chunks)


print("Number of chunks:", len(chunks))
print("Vector size:", len(vectors[0]))

print("\nFirst vector:")
print(vectors[0])