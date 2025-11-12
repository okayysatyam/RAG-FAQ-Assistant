import os
import pickle
import numpy as np
from typing import List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from .gemini_utils import get_embedding

# --- Configuration ---

load_dotenv()

USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Fallback to Sentence Transformers ---

try:
    # Ensure this package is installed via requirements.txt
    from sentence_transformers import SentenceTransformer, CrossEncoder
    model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except ImportError:
    model = None
    reranker = None

# --- FAISS Setup ---

import faiss

# Use the paths defined by your project structure

VSTORE_PATH = os.getenv("VECTOR_STORE_PATH", "backend/faiss.index")
METADATA_PATH = "backend/metadata.pkl"


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Return embeddings for a list of texts. Uses Gemini if USE_GEMINI True and set up,
    otherwise uses local sentence-transformers model.
    """
    if USE_GEMINI and GEMINI_API_KEY:
        embeddings = [get_embedding(text) for text in texts]
        return np.array(embeddings)
    elif model:
        return model.encode(texts)
    else:
        raise ValueError("No embedding model available")


def load_index() -> Tuple[faiss.Index, dict]:
    """
    Load FAISS index and metadata.
    """
    if not Path(VSTORE_PATH).exists():
        raise FileNotFoundError(f"FAISS index not found at {VSTORE_PATH}")
    if not Path(METADATA_PATH).exists():
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")

    index = faiss.read_index(VSTORE_PATH)
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(query: str, top_k=4, use_reranking=True) -> List[str]:
    """
    Search with optional re-ranking to refine context quality.
    Args:
        query: User query string
        top_k: Number of results to return after re-ranking
        use_reranking: Whether to apply re-ranking model
    """
    index, meta = load_index()

    # Step 1: Initial FAISS retrieval (get more candidates for re-ranking)
    initial_k = top_k * 3 if use_reranking else top_k
    q_vec = embed_texts([query])
    D, I = index.search(q_vec, initial_k)

    # Step 2: Collect candidate chunks
    candidates = []
    for idx in I[0]:
        if idx >= 0 and idx < len(meta["docs"]):
            _, chunk = meta["docs"][idx]
            candidates.append(chunk)

    # Step 3: Apply re-ranking if enabled and model available
    if use_reranking and reranker and len(candidates) > 0:
        pairs = [[query, chunk] for chunk in candidates]
        scores = reranker.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = [candidates[i] for i in ranked_indices]
        return results

    # Otherwise return initial FAISS results truncated to top_k
    return candidates[:top_k]

