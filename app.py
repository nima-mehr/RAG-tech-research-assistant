import os
import shutil
import time
import streamlit as st

from services.rag_engine import RAGEngine


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tech Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chunks" not in st.session_state:
    st.session_state.chunks = 0

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

engine = st.session_state.engine


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🤖 Tech Research Assistant")

st.caption(
    "A Retrieval-Augmented Generation (RAG) application for answering questions from PDF documents."
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
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
            key="process_btn",
            use_container_width=True
        ):

            start = time.time()

            with st.spinner("Creating embeddings..."):

                chunks = engine.process_pdf(pdf_path)

            processing_time = time.time() - start

            st.session_state.processed = True
            st.session_state.chunks = chunks
            st.session_state.current_pdf = uploaded_file.name

            st.success("✅ Document processed successfully!")

            st.caption(f"⏱ Processing time: {processing_time:.2f} seconds")

    if st.session_state.processed:

        st.metric(
            "Chunks Created",
            st.session_state.chunks
        )

    st.divider()

    st.subheader("📊 System Status")

    st.success("Engine Ready")

    if st.session_state.processed:
        st.success("Document Loaded")
    else:
        st.warning("No Document Loaded")

    st.divider()

    if st.button(
        "🗑 Clear Session",
        key="clear_btn",
        use_container_width=True
    ):

        st.session_state.processed = False
        st.session_state.chunks = 0
        st.session_state.current_pdf = None

        st.success("Session cleared.")


# --------------------------------------------------
# Main Area
# --------------------------------------------------
if st.session_state.processed:

    st.subheader("💬 Ask Questions")

    st.write(
        f"**Current document:** `{st.session_state.current_pdf}`"
    )

    question = st.text_input(
        "Enter your question"
    )

    if st.button(
        "Ask",
        key="ask_btn",
        type="primary"
    ):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            start = time.time()

            try:

                with st.spinner("Searching and generating answer..."):

                    result = engine.ask(question)

            except Exception as e:

                st.error(f"Error: {e}")
                st.stop()

            elapsed = time.time() - start

            with st.container(border=True):

                st.subheader("🤖 Answer")

                st.write(result["answer"])

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


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Built with ❤️ using Python • Streamlit • ChromaDB • SentenceTransformers • Ollama"
)