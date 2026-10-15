import os
import hashlib
import tempfile
from typing import List, Tuple

# Workaround for OpenMP duplicate runtime issues on some Windows/Anaconda setups
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


APP_TITLE = "Course Notes QA Assistant (RAG)"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_RULES = (
    "You are a course-notes question answering assistant.\n"
    "Use ONLY the provided context to answer.\n"
    "If the answer is not explicitly in the context, say: "
    "\"I don't know based on the provided notes.\"\n"
    "Do not use outside knowledge.\n"
    "Keep the answer concise and academic.\n"
)


def read_pdf_pages(file_bytes: bytes) -> List[Tuple[int, str]]:
    """Return list of (page_number_1based, page_text)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append((i, text))

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return pages


def pages_to_documents(pages: List[Tuple[int, str]], source_name: str) -> List[Document]:
    docs: List[Document] = []
    for page_num, text in pages:
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"source": source_name, "page": page_num}))
    return docs


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


def build_vectorstore(chunks: List[Document]) -> FAISS:
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def make_llm(groq_api_key: str, model_name: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
    )


def get_docs_compatible(retriever, question: str):
    """Compatible across LangChain versions."""
    if hasattr(retriever, "invoke"):
        return retriever.invoke(question)
    return retriever.get_relevant_documents(question)


def answer_question(llm: ChatGroq, vs: FAISS, question: str, k: int):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    retrieved_docs = get_docs_compatible(retriever, question)
    context = "\n\n".join([d.page_content for d in retrieved_docs])

    prompt = f"""{SYSTEM_RULES}

Context:
{context}

Question:
{question}

Answer:"""

    resp = llm.invoke(prompt)
    return resp.content, retrieved_docs


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Paste your Groq API key here. It will not be stored."
    )

    model_name = st.selectbox(
        "Groq Model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "qwen/qwen3-32b",
            "moonshotai/kimi-k2-instruct-0905",
            "groq/compound",
            "groq/compound-mini",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
        ],
        index=0,
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    k = st.slider("Top-k Chunks", 1, 8, 3, 1)

    st.markdown("---")
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk Size", 400, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, 10)

st.subheader("1) Upload PDF(s)")
uploaded_files = st.file_uploader(
    "Browse and upload course note PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

col1, col2 = st.columns([1, 2])
build_clicked = col1.button("Understand Docs", type="primary")
# Session state init
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "fingerprint" not in st.session_state:
    st.session_state.fingerprint = None
if "sources" not in st.session_state:
    st.session_state.sources = []

if build_clicked:
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Building index..."):
            combined_hash = hashlib.sha256()
            all_docs: List[Document] = []
            sources: List[str] = []

            for f in uploaded_files:
                file_bytes = f.getvalue()
                combined_hash.update(file_bytes)
                sources.append(f.name)

                pages = read_pdf_pages(file_bytes)
                docs = pages_to_documents(pages, source_name=f.name)
                all_docs.extend(docs)

            fingerprint = combined_hash.hexdigest()

            chunks = chunk_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vs = build_vectorstore(chunks)

            st.session_state.vectorstore = vs
            st.session_state.fingerprint = fingerprint
            st.session_state.sources = sources

        st.success(f"Index built for: {', '.join(st.session_state.sources)}")

st.divider()
st.subheader("2) Ask a Question")

question = st.text_input("Type your question about the uploaded PDFs")
ask_clicked = st.button("Get Answer")

if ask_clicked:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif st.session_state.vectorstore is None:
        st.error("Please build the index first (upload PDFs and click Understand Docs).")
    elif not question.strip():
        st.error("Please type a question.")
    else:
        llm = make_llm(groq_api_key=groq_api_key, model_name=model_name, temperature=temperature)

        with st.spinner("Retrieving and generating..."):
            answer, retrieved = answer_question(llm, st.session_state.vectorstore, question, k=k)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources")
        for i, d in enumerate(retrieved, start=1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            with st.expander(f"{i}) {src} — page {page}"):
                st.write(d.page_content)