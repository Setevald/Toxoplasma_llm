"""
FastAPI server for Toxoplasma LLM backend.

Responsibilities:
- Expose REST endpoints.
- Validate request payloads.
- Call model inference logic.
- Return JSON responses.

This file should NOT contain model logic.
"""
import os

os.environ["HF_HOME"] = "D:\\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "D:\\hf_cache"
os.environ["TORCH_HOME"] = "D:\\hf_cache"

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, generate_response
from retriever import build_index, retrieve, DEFAULT_DATA_DIR, chunk_text, extract_text_from_pdf
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend
    allow_credentials=True,
    allow_methods=["*"],  # THIS allows OPTIONS
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
def startup_event():
    """
    Runs when server starts.

    Loads:
    - LLM model
    - Vector search index
    """
    load_model()
    build_index()

@app.post("/chat")
def chat(request: ChatRequest):
    """
    Main Chat endpoint that handles incoming chat messages, retrieves relevant context, and generates a response.
    Args:
        request (ChatRequest): The incoming chat request containing the user's message.
    """
    print("[CHAT] Received:", request.message, flush=True)

    t0 = time.time()
    retrieved = retrieve(request.message)
    print(f"[CHAT] Retrieval done in {time.time() - t0:.2f}s", flush=True)

    context_texts = [item["text"] for item in retrieved]

    t1 = time.time()
    response = generate_response(
        request.message,
        context_chunks=context_texts
    )
    print(f"[CHAT] Generation done in {time.time() - t1:.2f}s", flush=True)

    return {
        "response": response,
        "sources": [
            f"{item['source']} (chunk {item['chunk_id']})"
            for item in retrieved
        ]
    }
    
@app.post("/upload-source")
async def upload_source(file: UploadFile = File(...)):
    """
    Upload a new source file and rebuild the FAISS index.

    Supported files:
    - .txt
    - .pdf

    Flow:
    1. Receive uploaded file.
    2. Save it into the data folder.
    3. If it is PDF, test whether text can be extracted.
    4. Rebuild FAISS index.
    5. Return upload status.
    """

    safe_filename = Path(file.filename).name
    lower_filename = safe_filename.lower()

    if not lower_filename.endswith((".txt", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported."
        )

    os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)

    save_path = os.path.join(DEFAULT_DATA_DIR, safe_filename)

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # TXT file handling
    if lower_filename.endswith(".txt"):
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Could not read TXT file. Please upload a UTF-8 encoded .txt file."
            )

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Uploaded TXT file has no readable text."
            )

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

        chunks_added = len(chunk_text(text))

    # PDF file handling
    elif lower_filename.endswith(".pdf"):
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        extracted_text = extract_text_from_pdf(save_path)

        if not extracted_text.strip():
            os.remove(save_path)

            raise HTTPException(
                status_code=400,
                detail="PDF uploaded, but no readable text could be extracted. It may be a scanned PDF."
            )

        chunks_added = len(chunk_text(extracted_text))

    # Rebuild FAISS so the newly uploaded source becomes searchable
    build_index()

    return {
        "message": "Source uploaded and indexed successfully.",
        "filename": safe_filename,
        "file_type": "pdf" if lower_filename.endswith(".pdf") else "txt",
        "chunks_added": chunks_added
    }