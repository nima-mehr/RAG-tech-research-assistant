import os

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PATH = os.getenv("CHROMA_PATH", "database")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
CHUNK_SIZE = _int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 80)
TOP_K = _int("TOP_K", 3)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "pdfs")
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 64)
