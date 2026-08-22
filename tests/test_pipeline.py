from pathlib import Path

from services.chunker import chunk_pages, chunk_text
from services.pdf_loader import load_pdf, load_pdf_pages


SAMPLE_PDF = Path("pdfs/sample.pdf")


def test_load_sample_pdf():
    text = load_pdf(str(SAMPLE_PDF))
    assert "WireGuard" in text or "VPN" in text or len(text) > 20


def test_load_pdf_pages_have_metadata():
    pages = load_pdf_pages(str(SAMPLE_PDF))
    assert pages
    assert all(page["page"] >= 1 for page in pages)
    assert all(page["source"] == "sample.pdf" for page in pages)
    assert all(page["text"].strip() for page in pages)


def test_chunk_text_splits_and_strips():
    text = "First paragraph.\n\nSecond paragraph is a bit longer. Third sentence follows."
    chunks = chunk_text(text, chunk_size=40, overlap=5)
    assert len(chunks) >= 1
    assert all(chunk == chunk.strip() for chunk in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_pages_keeps_page_numbers():
    pages = [
        {"source": "book.pdf", "page": 1, "text": "Alpha " * 40},
        {"source": "book.pdf", "page": 2, "text": "Beta " * 40},
    ]
    records = chunk_pages(pages, chunk_size=50, overlap=10)
    assert records
    assert {record["page"] for record in records} == {1, 2}
    assert all(record["source"] == "book.pdf" for record in records)
