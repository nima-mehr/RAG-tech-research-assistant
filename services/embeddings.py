from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(model_name or EMBEDDING_MODEL)

    def create_embeddings(self, texts: list[str], batch_size: int = 64):
        if not texts:
            return []
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
