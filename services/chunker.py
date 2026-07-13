from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    chunks = splitter.split_text(text)

    return chunks