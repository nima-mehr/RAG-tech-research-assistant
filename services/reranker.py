from collections.abc import Sequence

from config import RERANK_MODEL


class Reranker:
    """Cross-encoder scorer. Model is loaded on first use."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or RERANK_MODEL
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        raw = self._load().predict(pairs)
        return [float(value) for value in raw]
