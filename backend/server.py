import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests 

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Crucial Relative Imports ---

from .vectorstore import search
from .ingest import process_and_index_content
from .gemini_utils import generate_answer

# --- Load Config ---

load_dotenv()

USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- FastAPI Setup ---

limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour", "10/minute"])

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Middleware ---

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Local Generator if Gemini not used ---

local_generator = None

if not USE_GEMINI:
    try:
        from transformers import pipeline

        print("Loading local fallback model gpt2...")
        local_generator = pipeline(
            "text-generation",
            model="gpt2",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device="cpu",
        )
        print("Local model loaded successfully.")
    except ImportError:
        local_generator = None
        print("Could not load local model; transformers package not found.")


# --- Request Models ---

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4
    use_reranking: bool = True


@app.post("/query")
@limiter.limit("10/minute")
async def query(req: QueryRequest, request: Request):
    try:
        retrieved_chunks = search(req.question, top_k=req.top_k, use_reranking=req.use_reranking)
        context = "\n\n---\n\n".join(retrieved_chunks)
        prompt = f"Context: {context}\n\nQuestion: {req.question}\n\nAnswer in a clear and concise way."

        if USE_GEMINI and GEMINI_API_KEY:
            answer = generate_answer(prompt)
        elif local_generator:
            outputs = local_generator(prompt, max_length=200)
            answer = outputs[0]["generated_text"]
        else:
            raise HTTPException(status_code=503, detail="No LLM backend available")

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
@limiter.limit("5/hour")
async def ingest_document(file: UploadFile = File(...), request: Request = None):
    try:
        contents = await file.read()
        await process_and_index_content(contents.decode("utf-8"))
        return {"detail": "Document ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
