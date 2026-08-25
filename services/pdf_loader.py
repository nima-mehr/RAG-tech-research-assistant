from pathlib import Path

from pypdf import PdfReader


def load_pdf_document(file_path: str) -> dict:
    """Load a PDF with page-level text and basic document stats.

    Returns:
        {
            "source": filename,
            "total_pages": int,          # pages in the file
            "text_pages": int,           # pages with extractable text
            "pages": [{"source", "page", "text"}, ...],
        }
    """
    reader = PdfReader(file_path)
    source = Path(file_path).name
    total_pages = len(reader.pages)
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        text = extracted.strip()
        if text:
            pages.append({"source": source, "page": index, "text": text})

    return {
        "source": source,
        "total_pages": total_pages,
        "text_pages": len(pages),
        "pages": pages,
    }


def load_pdf_pages(file_path: str) -> list[dict]:
    """Extract text per page so large books keep page metadata."""
    return load_pdf_document(file_path)["pages"]


def load_pdf(file_path: str) -> str:
    """Return all extractable text joined with blank lines (tests / simple use)."""
    pages = load_pdf_pages(file_path)
    return "\n\n".join(page["text"] for page in pages)
