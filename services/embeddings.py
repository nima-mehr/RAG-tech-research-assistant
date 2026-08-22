from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(model_name or EMBEDDING_MODEL)

    def create_embeddings(self, texts: list[str]):
        if not texts:
            return []
        return self.model.encode(texts, show_progress_bar=False)
