__all__ = ["RAGEngine"]


def __getattr__(name: str):
    if name == "RAGEngine":
        from services.rag_engine import RAGEngine

        return RAGEngine
    raise AttributeError(name)
