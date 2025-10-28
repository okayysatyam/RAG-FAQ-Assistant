from pathlib import Path
from pdfminer.high_level import extract_text
from bs4 import BeautifulSoup
import re
from .vectorstore import add_or_create_faiss_index
from io import BytesIO # Import for handling file bytes

# --- Helper functions ---

def read_pdf_bytes(file_bytes):
    """Reads PDF content from bytes/stream instead of a file path."""
    try:
        # Use BytesIO to pass the bytes to extract_text
        txt = extract_text(BytesIO(file_bytes))
        return txt
    except Exception as e:
        print("PDF read error:", e)
        return ""

def clean_text(t):
    t = re.sub(r"\s+", " ", t).strip()
    return t


def chunk_text(text, chunk_size=800, overlap=100):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
# ------------------------


def process_and_index_content(filename: str, file_bytes: bytes):
    """
    Processes the raw bytes of an uploaded file and adds it to the FAISS index.
    """
    text = ""
    file_extension = Path(filename).suffix.lower()

    if file_extension in [".txt", ".md"]:
        text = file_bytes.decode("utf-8", errors="ignore")
    elif file_extension == ".pdf":
        text = read_pdf_bytes(file_bytes)
    elif file_extension in [".html", ".htm"]:
        # Decode first, then pass to BeautifulSoup
        raw = file_bytes.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "html.parser")
        text = soup.get_text()
    else:
        # Unsupported file type for RAG processing
        return False, f"Unsupported file type: {file_extension}"

    text = clean_text(text)
    if not text:
        return False, "File processed but contained no readable text."

    chunks = chunk_text(text)
    docs = []
    
    for idx, ch in enumerate(chunks):
        # The document ID format is crucial for tracing the source
        docs.append((f"{filename}_chunk_{idx}", ch))
    
    add_or_create_faiss_index(docs)
    
    return True, f"Successfully indexed {len(chunks)} chunks from {filename}."


def ingest_folder(folder="backend/sample_docs"):
    """
    Helper function for initial/manual indexing of the sample_docs folder.
    """
    print(f"--- Running initial ingestion from {folder} ---")
    all_chunks = []
    
    # Need to read content as bytes for consistency, but for this old function 
    # reading text from file path is simpler.
    def read_text_file(path):
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    def read_pdf_path(path):
        try:
            return extract_text(path)
        except Exception:
            return ""

    for p in Path(folder).iterdir():
        if p.is_file():
            text = ""
            if p.suffix.lower() in [".txt", ".md"]:
                text = read_text_file(p)
            elif p.suffix.lower() == ".pdf":
                text = read_pdf_path(p)
            elif p.suffix.lower() in [".html", ".htm"]:
                raw = read_text_file(p)
                soup = BeautifulSoup(raw, "html.parser")
                text = soup.get_text()
            
            text = clean_text(text)
            if not text: continue

            chunks = chunk_text(text)
            for idx, ch in enumerate(chunks):
                all_chunks.append((f"{p.name}_chunk_{idx}", ch))

    if all_chunks:
        add_or_create_faiss_index(all_chunks)
        print("--- Initial ingestion complete. ---")
    else:
        print("No documents found or processed in the sample_docs folder.")


if __name__ == "__main__":
    # Ensure this runs the initial ingestion correctly
    ingest_folder()
