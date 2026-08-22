import os
import time

import streamlit as st

from config import UPLOAD_DIR
from services.rag_engine import RAGEngine


st.set_page_config(
    page_title="Tech Research Assistant",
    page_icon="🤖",
    layout="wide",
)


def get_engine() -> RAGEngine:
    if "engine" not in st.session_state:
        st.session_state.engine = RAGEngine()
    return st.session_state.engine


def init_state() -> None:
    st.session_state.setdefault("last_added", 0)


init_state()
engine = get_engine()
library = engine.list_documents()
total_chunks = sum(library.values())

st.title("🤖 Tech Research Assistant")
st.caption("Ask questions across one large PDF or a whole document library.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Add a 400-page book or several papers. Text-based PDFs work best.",
    )
    replace_library = st.checkbox("Replace entire library", value=False)

    if uploaded_files:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        saved_paths = []
        for uploaded_file in uploaded_files:
            pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(pdf_path, "wb") as handle:
                handle.write(uploaded_file.getbuffer())
            saved_paths.append(pdf_path)
            st.write(f"- {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("Process documents", type="primary", use_container_width=True):
            started = time.time()
            try:
                with st.spinner("Extracting pages and creating embeddings..."):
                    chunks = engine.process_pdfs(saved_paths, replace=replace_library)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state.last_added = chunks
                st.success(f"Indexed {chunks} chunks from {len(saved_paths)} file(s).")
                st.caption(f"Processing time: {time.time() - started:.2f}s")
                st.rerun()

    st.divider()
    st.subheader("Library")
    if library:
        st.metric("Indexed files", len(library))
        st.metric("Total chunks", total_chunks)
        for name, count in library.items():
            col_name, col_btn = st.columns([3, 1])
            col_name.caption(f"{name} · {count} chunks")
            if col_btn.button("✕", key=f"rm-{name}", help=f"Remove {name}"):
                engine.remove_document(name)
                st.rerun()
    else:
        st.warning("No documents indexed")

    if st.button("Clear library", use_container_width=True):
        engine.reset()
        st.session_state.last_added = 0
        st.rerun()

if library:
    st.subheader("Ask a question")
    filter_source = st.selectbox(
        "Search in",
        options=["All documents", *library.keys()],
    )
    question = st.text_input("Question")
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            started = time.time()
            source = None if filter_source == "All documents" else filter_source
            try:
                with st.spinner("Searching and generating an answer..."):
                    result = engine.ask(question, source=source)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.subheader("Answer")
                st.write(result["answer"])
                st.caption(f"Response time: {time.time() - started:.2f}s")

                with st.expander("Retrieved context"):
                    for index, source_hit in enumerate(result["sources"], start=1):
                        label = source_hit.get("source") or "document"
                        heading = f"Source {index} — {label}"
                        page = source_hit.get("page")
                        chunk = source_hit.get("chunk")
                        if page is not None:
                            heading += f" · page {page}"
                        if chunk is not None:
                            heading += f" · chunk {chunk}"
                        st.markdown(f"**{heading}**")
                        st.code(source_hit["text"], language=None)
else:
    st.info("Upload one or more PDFs in the sidebar and process them to start asking questions.")
