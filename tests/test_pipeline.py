from pathlib import Path

from services.chunker import chunk_pages, chunk_text
from services.pdf_loader import load_pdf, load_pdf_document, load_pdf_pages
from services.retriever import combine_scores, hybrid_bonus, mmr_select, parse_chroma_results


SAMPLE_PDF = Path("pdfs/sample.pdf")


def test_load_sample_pdf():
    text = load_pdf(str(SAMPLE_PDF))
    assert "WireGuard" in text or "VPN" in text or len(text) > 20


def test_load_pdf_document_stats():
    document = load_pdf_document(str(SAMPLE_PDF))
    assert document["source"] == "sample.pdf"
    assert document["total_pages"] >= 1
    assert document["text_pages"] >= 1
    assert document["text_pages"] <= document["total_pages"]
    assert len(document["pages"]) == document["text_pages"]


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


def test_parse_chroma_results():
    raw = {
        "documents": [["alpha passage", "beta passage"]],
        "metadatas": [[{"source": "a.pdf", "page": 1, "chunk": 0}, {"source": "b.pdf", "page": 2, "chunk": 1}]],
        "distances": [[0.2, 0.8]],
        "embeddings": [[[1.0, 0.0], [0.0, 1.0]]],
    }
    hits = parse_chroma_results(raw)
    assert len(hits) == 2
    assert hits[0]["source"] == "a.pdf"
    assert hits[0]["distance"] == 0.2


def test_hybrid_bonus_rewards_term_overlap():
    query = "wireguard vpn handshake"
    relevant = "WireGuard uses a handshake to establish a VPN tunnel."
    unrelated = "The bakery opens at dawn and sells bread."
    assert hybrid_bonus(query, relevant) > hybrid_bonus(query, unrelated)


def test_combine_scores_prefers_rerank():
    hits = [
        {"text": "weak match but close vector", "distance": 0.1},
        {"text": "strong semantic answer about wireguard", "distance": 0.9},
    ]
    ranked = combine_scores(
        hits,
        rerank_scores=[0.1, 0.9],
        query="wireguard",
    )
    assert "wireguard" in ranked[0]["text"]
    assert ranked[0]["rerank_score"] == 0.9


def test_mmr_select_avoids_near_duplicates():
    hits = [
        {"text": "a", "score": 1.0, "embedding": [1.0, 0.0]},
        {"text": "a-copy", "score": 0.95, "embedding": [0.99, 0.01]},
        {"text": "different", "score": 0.7, "embedding": [0.0, 1.0]},
    ]
    selected = mmr_select(hits, top_k=2, lambda_mult=0.5)
    texts = {item["text"] for item in selected}
    assert "different" in texts
    assert len(selected) == 2
