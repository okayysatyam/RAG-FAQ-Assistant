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
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    model = None

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
        # Use the existing SDK function for each text
        embeddings = [get_embedding(text) for text in texts]
        return np.array(embeddings, dtype=np.float32)

    # fallback to the local model
    if model is None:
        raise EnvironmentError("Local embedding model not available. Install sentence-transformers or enable Gemini.")

    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype("float32")


def add_or_create_faiss_index(new_docs: List[Tuple[str, str]]):
    """
    Adds new documents to the FAISS index. Creates the index if it doesn't exist.
    docs: list of tuples (doc_id, text_chunk)
    """
    if not new_docs:
        print("No documents provided to index.")
        return

    texts = [t for _, t in new_docs]
    vecs = embed_texts(texts)
    
    if Path(VSTORE_PATH).exists() and Path(METADATA_PATH).exists():
        # Load existing index and metadata
        try:
            index = faiss.read_index(VSTORE_PATH)
            with open(METADATA_PATH, "rb") as f:
                meta = pickle.load(f)
            
            # Add vectors to index
            index.add(vecs)
            
            # Determine starting ID for consistency (although FAISS index is implicit)
            # We assume sequential naming in metadata.
            if meta["ids"]:
                last_id = max(meta["ids"])
                new_ids = list(range(last_id + 1, last_id + 1 + len(new_docs)))
            else:
                new_ids = list(range(len(new_docs)))

            # Update metadata
            meta["docs"].extend(new_docs)
            meta["ids"].extend(new_ids)
            
        except Exception as e:
            print(f"Error loading or updating index: {e}. Rebuilding from scratch.")
            # Fall through to the 'else' block to rebuild if corruption is detected.
            # In a real app, this should be handled more gracefully.
            return add_or_create_faiss_index(new_docs, force_rebuild=True)
            
    else:
        # Create a brand new index
        dim = vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vecs)
        
        # Create metadata from scratch
        ids = list(range(len(new_docs)))
        meta = {"ids": ids, "docs": new_docs}

    # Save the updated index and metadata
    faiss.write_index(index, VSTORE_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(meta, f)
    
    print(f"Indexed {len(new_docs)} new vectors. Total vectors: {index.ntotal}")

# The old 'build_faiss' is commented out/removed since it's now handled by the new function.
# If you run ingest.py directly, it will use the new function.


def load_index():
    if not Path(VSTORE_PATH).exists() or not Path(METADATA_PATH).exists():
        raise FileNotFoundError("Vector index not found. Run ingestion first or upload documents.")
    index = faiss.read_index(VSTORE_PATH)
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search(query: str, top_k=4):
    index, meta = load_index()
    q_vec = embed_texts([query])
    D, I = index.search(q_vec, top_k)
    results = []
    
    for idx in I[0]:
        if idx >= 0 and idx < len(meta["docs"]):
             _, chunk = meta["docs"][idx]
             results.append(chunk)
    return results
