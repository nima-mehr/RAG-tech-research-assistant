from services.pdf_loader import load_pdf
from services.chunker import chunk_text
from services.embeddings import EmbeddingModel
from services.vector_store import VectorStore


# Load PDF
text = load_pdf("pdfs/sample.pdf")


# Split text
chunks = chunk_text(
    text,
    chunk_size=200,
    overlap=40
)

# Create embeddings
embedding_model = EmbeddingModel()

vectors = embedding_model.create_embeddings(chunks)


# Store in Chroma
db = VectorStore()

db.add_documents(
    chunks,
    vectors
)


print("Documents stored successfully!")