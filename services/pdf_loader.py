from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """Return all extractable text joined with blank lines (tests / simple use)."""
    pages = load_pdf_pages(file_path)
    return "\n\n".join(page["text"] for page in pages)


def load_pdf_pages(file_path: str) -> list[dict]:
    """Extract text per page so large books keep page metadata."""
    reader = PdfReader(file_path)
    source = Path(file_path).name
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        text = extracted.strip()
        if text:
            pages.append({"source": source, "page": index, "text": text})

    return pages
