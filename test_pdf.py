from services.pdf_loader import load_pdf
from services.chunker import chunk_text

text = load_pdf("pdfs/sample.pdf")

chunks = chunk_text(text, chunk_size=50, overlap=10)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print("\n" + "-" * 50)
    print(f"Chunk {i+1}:")
    print(chunk)