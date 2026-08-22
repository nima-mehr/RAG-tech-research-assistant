from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    """Split each page separately so every chunk keeps source + page number."""
    records = []
    for page in pages:
        pieces = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
        for index, text in enumerate(pieces):
            records.append(
                {
                    "text": text,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": index,
                }
            )
    return records
