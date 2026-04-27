# Toxoplasma LLM Backend

This is the backend service for a Toxoplasma-focused RAG chatbot.

The system uses a Retrieval-Augmented Generation pipeline to answer questions based on provided Toxoplasma-related source documents. Source files are processed, split into chunks, converted into embeddings, stored in a FAISS index, and then retrieved when the user asks a question.

## Features

- Toxoplasma-focused chatbot backend
- RAG-based answering using source documents
- FAISS vector search for document retrieval
- Local LLM generation using Hugging Face Transformers
- Supports `.txt` and `.pdf` source files
- Upload new source documents through an API endpoint
- Returns retrieved source information with chatbot answers

## Project Structure

```txt
toxoplasma_llm/
├── app/
│   ├── main.py
│   ├── model.py
│   ├── retriever.py
│   └── config.py
├── requirements.txt
└── README.md
```
## Requirements

Recommended Python version:

```txt
Python 3.10+
```

The backend dependencies are listed in:

```txt
requirements.txt
```

Current dependencies:

```txt
fastapi
uvicorn[standard]
python-multipart
torch
transformers
accelerate
sentence-transformers
faiss-cpu
numpy
pypdf
```

### What Each Dependency Is Used For

`fastapi` is used to create the backend API.

`uvicorn[standard]` is used to run the FastAPI server locally.

`python-multipart` is required by FastAPI to accept uploaded files through `multipart/form-data`.

`torch` is used as the deep learning backend for running the language model.

`transformers` is used to load and run Hugging Face language models such as Qwen.

`accelerate` helps Hugging Face load models more efficiently, especially when using `device_map="auto"`.

`sentence-transformers` is used to create embeddings from document chunks and user questions.

`faiss-cpu` is used as the vector database/search index for retrieving relevant document chunks.

`numpy` is used for numerical operations, especially when handling embeddings.

`pypdf` is used to extract readable text from PDF source files.

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Go into the project folder:

```bash
cd toxoplasma_llm
```

Install all backend dependencies:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment, create and activate it first before installing dependencies.

Example using `venv` on Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Example using `venv` on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Backend

Go into the `app` folder:

```bash
cd app
```

Run the FastAPI server:

```bash
uvicorn main:app --reload
```

The backend will run at:

```txt
http://127.0.0.1:8000
```

The API documentation can be accessed at:

```txt
http://127.0.0.1:8000/docs
```

---

## Source Documents

The backend uses source documents to build the FAISS index.

Supported source file types:

```txt
.txt
.pdf
```

For PDF files, the backend extracts readable text from the PDF before adding it into the FAISS index.

Important note:

```txt
PDF extraction works best with normal text-based PDFs.
Scanned/image-only PDFs may not work because they require OCR.
```

If the PDF does not contain selectable text, the backend may not be able to extract useful content from it.

---

## API Endpoints

### 1. Chat Endpoint

```txt
POST /chat
```

Used to ask a question to the chatbot.

Example request body:

```json
{
  "message": "How is Toxoplasma gondii transmitted to humans?"
}
```

Example response:

```json
{
  "response": "Toxoplasma gondii can be transmitted through contaminated food, undercooked meat, contaminated water, and exposure to oocysts from cat feces.",
  "sources": [
    "Toxoplasma_00.pdf (chunk 3)",
    "Toxoplasma_01.pdf (chunk 5)"
  ]
}
```

---

### 2. Upload Source Endpoint

```txt
POST /upload-source
```

Used to upload a new source file into the knowledge base.

Request type:

```txt
multipart/form-data
```

Field name:

```txt
file
```

Supported file types:

```txt
.txt
.pdf
```

Example response:

```json
{
  "message": "Source uploaded and indexed successfully.",
  "filename": "new_toxoplasma_paper.pdf",
  "file_type": "pdf",
  "chunks_added": 20
}
```

After uploading a new source, the backend rebuilds the FAISS index so the chatbot can use the new document.

---

## How the RAG System Works

The backend follows this flow:

```txt
Source document
→ text extraction
→ text chunking
→ embedding generation
→ FAISS index storage
→ user question
→ retrieve relevant chunks
→ send chunks to LLM
→ generate answer
```

The LLM does not directly search the document by itself. Instead, the backend retrieves relevant document chunks first, then gives those chunks to the LLM as context.

---

## How the System Reduces Hallucination

The system reduces hallucination by grounding the model's answer in retrieved source chunks.

The general process is:

```txt
User question
→ retrieve relevant source chunks
→ include retrieved chunks in the prompt
→ instruct the model to answer using only the provided context
→ return the generated answer with source references
```

This does not guarantee that every answer is always correct, but it makes the answer more traceable because the response is based on retrieved documents.

For better reliability, the source documents should come from trusted academic, medical, or research-based references.

---

## Model Configuration

The model can be changed in:

```txt
app/config.py
```

Recommended model options:

```txt
Qwen/Qwen2.5-0.5B-Instruct  -> fastest for testing
Qwen/Qwen2.5-1.5B-Instruct  -> better demo quality
Qwen/Qwen2.5-7B-Instruct    -> better quality but requires stronger GPU/server
```

For local demo testing, the recommended model is:

```txt
Qwen/Qwen2.5-0.5B-Instruct
```

or:

```txt
Qwen/Qwen2.5-1.5B-Instruct
```

The `7B` model is heavier and is more suitable for a machine with stronger GPU support.

---

## Performance Notes

The retrieval step is usually fast because FAISS search is lightweight for small datasets.

The slower part is usually the LLM generation step, especially when running larger models locally.

Example:

```txt
Retrieval: less than 1 second
Generation: can take several seconds or longer depending on model and hardware
```

If generation is too slow, use a smaller model such as:

```txt
Qwen/Qwen2.5-0.5B-Instruct
```

or reduce:

```txt
MAX_NEW_TOKENS
```

inside `app/config.py`.

---

## Known Limitations

- Response generation can be slow depending on the selected model and hardware.
- Larger models may require a GPU.
- PDF support only works properly for text-based PDFs.
- Scanned PDFs may not be readable without OCR.
- The chatbot should only be treated as a research prototype, not a medical diagnosis tool.
- The accuracy of the answer depends on the quality of the provided source documents.
- The system reduces hallucination by using retrieved context, but it cannot fully guarantee that every generated answer is correct.

---

## Notes

This backend is designed as a prototype for a Toxoplasma research chatbot. It is meant to demonstrate how RAG can be used to ground LLM answers using research documents.

For best results, use trusted academic or medical sources as input documents.
