from collections.abc import Sequence

import numpy as np

from services.reranker import Reranker


def parse_chroma_results(results: dict) -> list[dict]:
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    embeddings = (results.get("embeddings") or [[]])[0]

    hits: list[dict] = []
    for index, text in enumerate(documents):
        meta = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        embedding = embeddings[index] if index < len(embeddings) else None
        hits.append(
            {
                "text": text,
                "source": meta.get("source") or "document",
                "page": meta.get("page"),
                "chunk": meta.get("chunk"),
                "distance": float(distance) if distance is not None else None,
                "embedding": embedding,
            }
        )
    return hits


def hybrid_bonus(query: str, text: str) -> float:
    """Small lexical overlap bonus so exact terms survive embedding misses."""
    query_terms = {token for token in _tokens(query) if len(token) > 2}
    if not query_terms:
        return 0.0
    text_terms = set(_tokens(text))
    overlap = len(query_terms & text_terms) / len(query_terms)
    return overlap


def _tokens(text: str) -> list[str]:
    return [part.lower() for part in "".join(
        ch.lower() if ch.isalnum() else " " for ch in text
    ).split() if part]


def combine_scores(
    hits: list[dict],
    rerank_scores: Sequence[float] | None = None,
    lexical_weight: float = 0.15,
    query: str = "",
) -> list[dict]:
    ranked = []
    for index, hit in enumerate(hits):
        item = dict(hit)
        distance = item.get("distance")
        vector_score = 1.0 / (1.0 + distance) if distance is not None else 0.0
        item["vector_score"] = vector_score

        rerank = None
        if rerank_scores is not None and index < len(rerank_scores):
            rerank = float(rerank_scores[index])
        item["rerank_score"] = rerank

        lexical = hybrid_bonus(query, item.get("text") or "") if query else 0.0
        item["lexical_score"] = lexical

        if rerank is not None:
            # Cross-encoder dominates; lexical is a light tie-breaker.
            item["score"] = rerank + lexical_weight * lexical
        else:
            item["score"] = vector_score + lexical_weight * lexical
        ranked.append(item)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def mmr_select(
    hits: list[dict],
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[dict]:
    """Pick diverse passages among already-scored hits when embeddings exist."""
    if len(hits) <= top_k:
        return hits[:top_k]

    usable = [hit for hit in hits if hit.get("embedding") is not None]
    if len(usable) < 2:
        return hits[:top_k]

    embeddings = np.asarray([hit["embedding"] for hit in usable], dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    embeddings = embeddings / norms

    scores = np.asarray([hit["score"] for hit in usable], dtype=float)
    score_min, score_max = float(scores.min()), float(scores.max())
    if score_max - score_min < 1e-9:
        norm_scores = np.ones_like(scores)
    else:
        norm_scores = (scores - score_min) / (score_max - score_min)

    selected: list[int] = []
    remaining = list(range(len(usable)))
    first = int(np.argmax(norm_scores))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < top_k:
        selected_matrix = embeddings[selected]
        best_index = remaining[0]
        best_value = -1e9
        for candidate in remaining:
            relevance = float(norm_scores[candidate])
            similarity = float(np.max(embeddings[candidate] @ selected_matrix.T))
            value = lambda_mult * relevance - (1.0 - lambda_mult) * similarity
            if value > best_value:
                best_value = value
                best_index = candidate
        selected.append(best_index)
        remaining.remove(best_index)

    return [usable[index] for index in selected]


def retrieve_hits(
    *,
    db,
    embedding_model,
    question: str,
    top_k: int,
    candidate_k: int,
    source: str | None = None,
    reranker: Reranker | None = None,
    use_rerank: bool = True,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
) -> list[dict]:
    library_size = db.count()
    fetch_k = candidate_k if use_rerank or use_mmr else top_k
    fetch_k = max(top_k, min(fetch_k, library_size))

    query_embedding = embedding_model.create_embeddings([question])[0]
    results = db.search(
        query_embedding,
        results=fetch_k,
        where={"source": source} if source else None,
        include_embeddings=use_mmr,
    )
    hits = parse_chroma_results(results)
    if not hits:
        return []

    rerank_scores = None
    if use_rerank and reranker is not None:
        rerank_scores = reranker.score(question, [hit["text"] for hit in hits])

    ranked = combine_scores(hits, rerank_scores=rerank_scores, query=question)
    if use_mmr:
        return mmr_select(ranked, top_k=top_k, lambda_mult=mmr_lambda)
    return ranked[:top_k]
