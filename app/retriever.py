"""
Retriever module for RAG (Retrieval-Augmented Generation).

Responsibilities:
- Load documents from disk
- Chunk documents into manageable passages
- Generate vector embeddings
- Build FAISS similarity index
- Retrieve top-k relevant chunks for a query

This module is intentionally independent of the LLM.
It handles only document memory and retrieval logic.
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")

# Embedding model (lightweight and strong for semantic similarity)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Global state (kept simple for MVP)
index = None
documents = []
metadata = []  # stores source tracking (filename + chunk id)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract readable text from a PDF file.

    Important:
    - This works for normal PDFs with selectable text.
    - This will NOT work well for scanned/image-only PDFs.
    - Scanned PDFs need OCR, which is a different feature.
    """

    reader = PdfReader(pdf_path)
    extracted_pages = []

    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text()

        if page_text:
            extracted_pages.append(page_text)
        else:
            print(f"[WARNING] No text extracted from page {page_number + 1} in {pdf_path}")

    return "\n\n".join(extracted_pages)

def load_documents(data_folder=DEFAULT_DATA_DIR):
    """
    Load documents from the data folder.

    Supported formats:
    - .txt
    - .pdf

    Returns:
        list[tuple[str, str]]:
            List of (filename, document_text).
    """

    docs = []

    if not os.path.exists(data_folder):
        print(f"[WARNING] Data folder '{data_folder}' not found.")
        return docs

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        lower_filename = filename.lower()

        try:
            # Load normal text files
            if lower_filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                if text.strip():
                    docs.append((filename, text))
                else:
                    print(f"[WARNING] Skipping empty TXT file: {filename}")

            # Load PDF files by extracting their text
            elif lower_filename.endswith(".pdf"):
                print(f"[INFO] Extracting PDF text: {filename}")
                text = extract_text_from_pdf(file_path)

                if text.strip():
                    docs.append((filename, text))
                else:
                    print(f"[WARNING] Skipping PDF with no extractable text: {filename}")

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")

    return docs


def chunk_text(text, chunk_size=300):
    """
    Split text into smaller word-based chunks.

    Args:
        text (str): Full document text.
        chunk_size (int): Number of words per chunk.

    Returns:
        list[str]: List of text chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def build_index(data_folder=DEFAULT_DATA_DIR):
    """
    Build FAISS index from dataset.

    This:
    - Loads documents
    - Chunks them
    - Embeds chunks
    - Builds similarity search index

    Must be called once at server startup.
    """
    global index, documents, metadata

    raw_docs = load_documents(data_folder)

    if not raw_docs:
        print("[WARNING] No documents found. Index not built.")
        return

    documents = []
    metadata = []

    # Chunk each document and track metadata
    for filename, doc_text in raw_docs:
        chunks = chunk_text(doc_text)

        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({
                "source": filename,
                "chunk_id": idx
            })

    # Generate embeddings
    embeddings = embedding_model.encode(documents, convert_to_numpy=True)

    # Normalize embeddings (important for cosine similarity behavior)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Use inner product index (works well with normalized embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"[INFO] FAISS index built with {len(documents)} chunks.")


def retrieve(query, top_k=3):
    """
    Retrieve top-k relevant chunks for a query.

    Args:
        query (str): User question.
        top_k (int): Number of chunks to retrieve.

    Returns:
        list[dict]:
            List of retrieved items:
            {
                "text": chunk_text,
                "source": filename,
                "chunk_id": id
            }
    """
    global index

    if index is None:
        raise RuntimeError("FAISS index is not built. Call build_index() first.")

    # Embed query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # Normalize query embedding
    faiss.normalize_L2(query_embedding)

    # Search
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        results.append({
            "text": documents[i],
            "source": metadata[i]["source"],
            "chunk_id": metadata[i]["chunk_id"]
        })

    return results
