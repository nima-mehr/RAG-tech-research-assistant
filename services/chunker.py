from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=500, overlap=100):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    return splitter.split_text(text)