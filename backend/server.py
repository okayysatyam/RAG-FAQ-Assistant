import torch 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests 

# --- Crucial Relative Imports ---
# These imports MUST be relative to work when run as a package
from .vectorstore import search 
from .ingest import process_and_index_content 
from .gemini_utils import generate_answer 

# --- FastAPI Setup ---
load_dotenv()
app = FastAPI()

# --- CORS Middleware ---
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Config / Model Loading ---
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load the local model ONCE at startup if Gemini is not used
local_generator = None
if not USE_GEMINI:
    try:
        from transformers import pipeline
        print("Loading local fallback model (gpt2)...")
        local_generator = pipeline(
            "text-generation",
            model="gpt2",
            trust_remote_code=True,
            torch_dtype=torch.float32, 
            device="cpu",
        )
        print("Local model loaded successfully.")
    except Exception as e:
        local_generator = None
        print(f"CRITICAL: Failed to load local model: {e}")
        print("The server will run, but local generation will fail.")


# --- Pydantic Model ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


# --- NEW Endpoint for Document Upload/Ingestion ---
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Accepts an uploaded file, processes it, and adds chunks to the vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        file_bytes = await file.read()
        success, message = process_and_index_content(file.filename, file_bytes)
        
        if success:
            return {"status": "success", "message": message, "filename": file.filename}
        else:
            raise HTTPException(status_code=400, detail=f"Ingestion failed: {message}")

    except Exception as e:
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error during ingestion: {e}")


# --- Existing Query Endpoint (NOW CORRECTED) ---
@app.post("/query")
def query(req: QueryRequest):
    # 1. Retrieve context
    try:
        retrieved_chunks = search(req.question, top_k=req.top_k)
        context = "\n\n".join(retrieved_chunks)
    except FileNotFoundError:
        return {"answer": "The knowledge base is empty. Please upload documents first.", "sources": []}
    except Exception as e:
        return {"answer": f"Error during document search: {e}", "sources": []}

    # 2. Build the prompt
    prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion:\n{req.question}\n\nAnswer:"

    # 3. Generate the answer
    reply = "Could not generate a response."

    # --- THIS ENTIRE BLOCK IS NOW CORRECTLY PLACED *INSIDE* THE 'query' FUNCTION ---
    if USE_GEMINI and GEMINI_API_KEY:
        # This code block executes *only* when the /query endpoint is called
        rag_prompt = f"Context: {context}\n\nQuestion: {req.question}\n\nAnswer:"
        
        try:
            # We call the 'generate_answer' function imported at the top
            reply = generate_answer(rag_prompt)
        except Exception as e:
            reply = f"Error calling Gemini API: {e}. Check API key and quota."
            
    else:
        # Use the local fallback model
        if local_generator:
            try:
                # Use the 'local_generator' model loaded at startup
                out = local_generator(prompt, max_length=256, num_return_sequences=1)
                if out and out[0]:
                    generated_text = out[0]["generated_text"]
                    reply = generated_text[len(prompt) :].strip()
            except Exception as e:
                reply = f"Error during local model generation: {e}"
        else:
            reply = "Local model is unavailable. Check startup logs for errors."
    # --- END OF MOVED BLOCK ---

    # 4. Return the final response
    return {"answer": reply, "sources": retrieved_chunks}