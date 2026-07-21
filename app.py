import os
import shutil
import time
import streamlit as st

from services.rag_engine import RAGEngine


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Tech Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()

if "processed" not in st.session_state:
    st.session_state.processed = False

engine = st.session_state.engine

# -----------------------------
# Header
# -----------------------------
st.title("🤖 Tech Research Assistant")
st.caption(
    "A Retrieval-Augmented Generation (RAG) system for answering questions from PDF documents."
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file:

        os.makedirs("pdfs", exist_ok=True)

        pdf_path = os.path.join(
            "pdfs",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        st.success("PDF uploaded successfully!")

        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

        if st.button(
            "⚙️ Process Document",
            use_container_width=True
        ):

            with st.spinner("Creating embeddings..."):

                chunks = engine.process_pdf(pdf_path)

            st.session_state.processed = True

            st.success(
                f"✅ Processing complete!\n\nChunks created: {chunks}"
            )

    st.divider()

    if st.button(
        "🗑 Clear Session",
        use_container_width=True
    ):

        st.session_state.processed = False

        st.success("Session cleared.")

# -----------------------------
# Main Area
# -----------------------------
if st.session_state.processed:

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question"
    )

    if st.button(
        "Ask",
        type="primary"
    ):

        if question.strip():

            start = time.time()

            with st.spinner("Searching and generating answer..."):

                result = engine.ask(question)

            elapsed = time.time() - start

            st.subheader("🤖 Answer")

            st.info(result["answer"])

            st.caption(
                f"⏱ Response time: {elapsed:.2f} seconds"
            )

            with st.expander(
                "📚 Retrieved Context",
                expanded=False
            ):

                for i, source in enumerate(
                    result["sources"],
                    start=1
                ):

                    st.markdown(f"### Source {i}")

                    st.code(
                        source,
                        language=None
                    )

else:

    st.info(
        "👈 Upload a PDF from the sidebar and click **Process Document** to begin."
    )