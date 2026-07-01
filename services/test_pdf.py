from services.pdf_loader import load_pdf

text = load_pdf("pdfs/sample.pdf")

print(text[:1000])