"""
rag.py
Everything related to Retrieval-Augmented Generation lives here:
loading PDFs, chunking, embedding, storing in Chroma, and retrieving relevant chunks.

Kept separate from backend.py (chat logic) and app.py (UI) so each file has ONE job.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free


def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Loads a PDF and splits it into chunks.

    chunk_size / chunk_overlap are deliberately exposed as params —
    experimenting with these is exactly the kind of thing worth noting
    in your project write-up (how did chunk size affect answer quality?).
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(pages)
    return chunks


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(chunks, persist_directory: str = PERSIST_DIR):
    """
    Embeds chunks and stores them in a local Chroma DB.
    Re-running this with new documents will add to the existing store
    unless you clear persist_directory first.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    return vectorstore


def load_vectorstore(persist_directory: str = PERSIST_DIR):
    """Loads an existing Chroma DB from disk (no re-embedding needed)."""
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


def get_retriever(vectorstore, k: int = 3):
    """k = how many chunks to retrieve per query. Start with 3, tune later."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_context(retriever, query: str) -> str:
    """
    Runs retrieval and formats results into a single context string
    ready to inject into the LLM prompt. Also returns which chunks were
    used, which matters later for evaluation (Layer 5).
    """
    docs = retriever.invoke(query)
    context = "\n\n".join(
        f"[Source: page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )
    return context, docs