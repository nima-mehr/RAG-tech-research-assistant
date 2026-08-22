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
    st.session_state.setdefault("processed", False)
    st.session_state.setdefault("chunks", 0)
    st.session_state.setdefault("current_pdf", None)


init_state()
engine = get_engine()

st.title("🤖 Tech Research Assistant")
st.caption("Ask questions about a PDF using local retrieval-augmented generation.")

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as handle:
            handle.write(uploaded_file.getbuffer())

        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

        if st.button("Process document", type="primary", use_container_width=True):
            started = time.time()
            try:
                with st.spinner("Creating embeddings..."):
                    chunks = engine.process_pdf(pdf_path, replace=True)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state.processed = True
                st.session_state.chunks = chunks
                st.session_state.current_pdf = uploaded_file.name
                st.success("Document processed.")
                st.caption(f"Processing time: {time.time() - started:.2f}s")

    if st.session_state.processed:
        st.metric("Chunks created", st.session_state.chunks)

    st.divider()
    st.subheader("Status")
    if st.session_state.processed:
        st.success("Document loaded")
    else:
        st.warning("No document loaded")

    if st.button("Clear session", use_container_width=True):
        engine.reset()
        st.session_state.processed = False
        st.session_state.chunks = 0
        st.session_state.current_pdf = None
        st.rerun()

if st.session_state.processed:
    st.subheader("Ask a question")
    st.write(f"**Current document:** `{st.session_state.current_pdf}`")

    question = st.text_input("Question")
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            started = time.time()
            try:
                with st.spinner("Searching and generating an answer..."):
                    result = engine.ask(question)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.subheader("Answer")
                st.write(result["answer"])
                st.caption(f"Response time: {time.time() - started:.2f}s")

                with st.expander("Retrieved context"):
                    for index, source in enumerate(result["sources"], start=1):
                        label = source.get("source") or "document"
                        chunk = source.get("chunk")
                        heading = f"Source {index} — {label}"
                        if chunk is not None:
                            heading += f" (chunk {chunk})"
                        st.markdown(f"**{heading}**")
                        st.code(source["text"], language=None)
else:
    st.info("Upload a PDF in the sidebar and process it to start asking questions.")
