import streamlit as st
import requests
from PIL import Image
import os
import io

# --- Configuration ---
BACKEND_URL = "http://localhost:8000"
QUERY_ENDPOINT = f"{BACKEND_URL}/query"
INGEST_ENDPOINT = f"{BACKEND_URL}/ingest"

st.set_page_config(page_title="RAG FAQ Assistant", layout="centered")

# --- Styling (Kept from original) ---
st.markdown(
    """
    <style>
        /* Page background */
        .stApp {
            background-color: #F7EED6;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #2E2E2E !important;
        }

        /* Input box */
        .stTextInput input {
            background-color: #2E2E2E !important;
            color: #FFFFFF !important;
            border-radius: 8px;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #D4A85C !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5em 1em !important;
            font-weight: 600 !important;
        }

        /* Sidebar / FAQ boxes */
        .faq-box, .faq-card {
            background-color: #E6C791;
            color: #2E2E2E;
            padding: 10px 15px;
            border-radius: 12px;
            margin-bottom: 8px;
            font-weight: 500;
        }

        /* Rating star section */
        .rating-star {
            color: #D4A85C !important;
            font-size: 1.5em;
        }

        /* Label text (Rate Answer, etc.) */
        label, p, span, div {
            color: #3B3B3B !important;
        }

        /* Chat message bubbles */
        .chat-box {
            background-color: #FFF7E9;
            border-radius: 15px;
            padding: 10px;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# --- File Upload and Ingestion Logic (Keep as-is) ---
def handle_upload():
    """Handles file upload via Streamlit and sends it to the FastAPI /ingest endpoint."""
    st.sidebar.markdown("## 📚 Add Documents")
    
    file_types = ["pdf", "txt", "md", "html"]
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF, TXT, MD, or HTML file to index:", 
        type=file_types
    )

    if uploaded_file is not None:
        if st.sidebar.button("Index Document"):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    response = requests.post(INGEST_ENDPOINT, files=files, timeout=120)
                    response.raise_for_status()
                    resp_json = response.json()
                    st.sidebar.success(f"Success: {resp_json['message']}")
                except Exception as e:
                    st.sidebar.error(f"Ingestion Error: {e}")

# ========================================================
# ---  START OF CORRECTED LAYOUT ---
# ========================================================

# CALL THE UPLOAD FUNCTION - It will automatically go to the sidebar
handle_upload()

# --- Main Application Layout ---
col1, col2 = st.columns([3,1])

with col1:
    # --- This is the main chat column ---
    st.markdown("<div style='display:flex; align-items:center; gap:20px'>", unsafe_allow_html=True)
    try:
        robot_img = Image.open("frontend/assets/robot.png")
        st.image(robot_img, width=90)
    except Exception:
        pass # Ignore if image not found
    st.markdown("<h2 style='margin:0;'>Ask a Question</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    query = st.text_input("Your Question", placeholder="Ask a question...", key="input", label_visibility="collapsed")
    
    if st.button("Ask"):
        if not query.strip():
            st.warning("Type something first.")
        else:
            with st.spinner("Thinking..."):
                payload = {"question": query, "top_k": 4}
                try:
                    resp = requests.post(QUERY_ENDPOINT, json=payload, timeout=30).json()
                    answer = resp.get("answer", "No answer returned.")
                    sources = resp.get("sources", [])
                except Exception as e:
                    answer = "Error contacting backend: " + str(e)
                    sources = []
            
            # ... (Rest of your answer/sources display code) ...
            st.markdown(f"<div class='chat-box'><strong>AI Assistant</strong><p>{answer}</p></div>", unsafe_allow_html=True)
            if sources and "knowledge base is empty" not in answer:
                 st.markdown("<div style='margin-top:10px; padding:10px; border-radius:12px; background:#fff'>", unsafe_allow_html=True)
                 st.write("### Referenced Passages")
                 for i, s in enumerate(sources):
                     st.write(f"**Chunk {i+1}:**", s[:300] + "...")
                 st.markdown("</div>", unsafe_allow_html=True)

    # ... (Rest of your feedback/rating code) ...
    st.write("#### Rate answer")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        if st.button("⭐"):
            st.success("Thanks! Recorded 1 star (local only).")


with col2:
    # --- This is the FAQ column ---
    st.markdown("## Related FAQs")
    faqs = [
        "How to upload documents?",
        "How does retrieval work?",
        "How do I add new docs?",
    ]
    for f in faqs:
        st.markdown(f"<div class='faq-card'>{f}</div>", unsafe_allow_html=True)
    
    mode = st.checkbox("Dark Mode")
    if mode:
        st.info("Dark mode enabled (visual only).")
