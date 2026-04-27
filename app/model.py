"""
Model module for Toxoplasma LLM Backend.

Responsibilities:
- Load tokenizer and model once at server startup
- Clean retrieved context before prompt injection
- Generate grounded responses using RAG
- Keep generation settings centralized via config.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_ID, MAX_NEW_TOKENS, TEMPERATURE, TOP_P

# Global objects so the model is loaded only once
tokenizer = None
model = None


def load_model():
    """
    Load tokenizer and model into memory.

    This function is called once during FastAPI startup.
    Loading globally avoids reloading the model on every request,
    which would be extremely slow and memory-inefficient.
    """
    global tokenizer, model

    # Prevent duplicate loading if startup is triggered again
    if tokenizer is not None and model is not None:
        print("Model already loaded. Skipping reload.")
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Model successfully loaded.")


def clean_context_text(text: str) -> str:
    """
    Clean raw text extracted from PDFs before sending it to the model.

    Why this is needed:
    - PDF-to-text conversion often produces noisy line breaks
    - extra whitespace can reduce readability
    - cleaning improves prompt quality and response quality

    Args:
        text (str): Raw retrieved chunk text

    Returns:
        str: Cleaned text
    """
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def generate_response(user_message: str, context_chunks=None) -> str:
    """
    Generate a response using retrieved context when available.

    Flow:
    1. Clean retrieved chunks
    2. Build a structured prompt
    3. Tokenize and run inference
    4. Decode and clean output

    Args:
        user_message (str):
            The user question.

        context_chunks (list[str] | None):
            Retrieved text chunks from the retriever module.

    Returns:
        str:
            Final generated answer.
    """
    global tokenizer, model

    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    context_text = ""

    # If RAG context exists, clean and join it
    if context_chunks and len(context_chunks) > 0:
        cleaned_chunks = []

        for chunk in context_chunks:
            cleaned_chunk = clean_context_text(chunk)
            cleaned_chunks.append(cleaned_chunk)

        context_text = "\n\n---\n\n".join(cleaned_chunks)

        prompt = (
            "You are a research assistant specialized in Toxoplasma gondii.\n"
            "Answer ONLY using the provided context.\n"
            "Do NOT use outside knowledge.\n"
            "If the answer is not found in the context, say: "
            "'Not found in provided sources.'\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Summarize the information instead of copying sentences.\n"
            "- Use 3 short bullet points.\n"
            "- Each bullet point must be one sentence only.\n"
            "- Do not add extra explanation after the bullet points.\n"
            "- Focus only on the most relevant medical facts.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question:\n{user_message}\n\n"
            "Answer:"
        )
    else:
        # Fallback mode if no retrieval context is available
        prompt = (
            "You are a research assistant specialized in Toxoplasma gondii.\n"
            f"User Question:\n{user_message}\n\n"
            "Answer:"
        )

    # Convert prompt to tensors and move them to the model device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Run inference without gradients
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode only the newly generated tokens (exclude the prompt)
    new_tokens = output_ids[0][inputs.input_ids.shape[-1]:]
    
    # Decode the generated tokens to text
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Final cleanup
    generated_text = generated_text.strip()
    generated_text = generated_text.replace("•", "-")

    return generated_text
