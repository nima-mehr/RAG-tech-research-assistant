from pathlib import Path

from services.chunker import chunk_text
from services.pdf_loader import load_pdf


SAMPLE_PDF = Path("pdfs/sample.pdf")


def test_load_sample_pdf():
    text = load_pdf(str(SAMPLE_PDF))
    assert "WireGuard" in text or "VPN" in text or len(text) > 20


def test_chunk_text_splits_and_strips():
    text = "First paragraph.\n\nSecond paragraph is a bit longer. Third sentence follows."
    chunks = chunk_text(text, chunk_size=40, overlap=5)
    assert len(chunks) >= 1
    assert all(chunk == chunk.strip() for chunk in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
