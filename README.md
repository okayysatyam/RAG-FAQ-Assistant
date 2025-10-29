# RAG FAQ Assistant

RAG FAQ Assistant is a Retrieval-Augmented Generation (RAG) platform designed to answer questions based on custom, user-uploaded documents. 
It leverages a Python FastAPI backend for AI processing and a Streamlit frontend for user interaction, providing accurate, context-aware 
responses grounded in the provided knowledge base.

---

## Table of Contents

* [Features](#features)
* [Getting Started](#getting-started)
* [Tech Stack](#tech-stack)
* [Project Structure](#project-structure)
* [Known Issues](#known-issues)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgements](#acknowledgements)

---

## Features

* **Retrieval-Augmented Generation (RAG) Pipeline:** Provides answers grounded in specific documents, minimizing hallucinations.
* **Dynamic Document Upload:** Users can upload new documents (PDF, TXT, MD, HTML) directly through the UI, which are automatically indexed and made available for querying.
* **Efficient Vector Search:** Utilizes FAISS for fast and accurate similarity search to find relevant document chunks.
* **Decoupled Frontend/Backend:** A modern architecture with a Streamlit frontend and FastAPI backend for scalability and maintainability.
* **Flexible LLM Integration:** Supports both the Google Gemini API (for high-quality generation) and a local fallback model (like GPT-2 or Phi-3) using Hugging Face Transformers, configurable via environment variables.
* **Real-time Indexing:** Uploaded documents are processed and added to the vector store on the fly.

---

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

1.  **Python 3.10+**
2.  **Git** installed on your system.
3.  **(Optional but Recommended)** A Google Gemini API Key obtained from [Google AI Studio](https://aistudio.google.com/app/apikey) for higher-quality answers and embeddings.

### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/okayysatyam/RAG-FAQ-Assistant.git
    cd RAG-FAQ-Assistant
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    * Copy the example environment file:
        ```bash
        copy .env.example .env
        # On macOS/Linux: cp .env.example .env
        ```
    * Open the newly created `.env` file.
    * **Paste your Google Gemini API Key** if you plan to use it (`GEMINI_API_KEY=YOUR_KEY_HERE`).
    * Set `USE_GEMINI` to `"true"` to use the Gemini API or `"false"` to use the local models (embedding and generation).

5.  **Build Initial Vector Store (Ingestion):**
    This step processes any documents in the `backend/sample_docs` folder and creates the initial `faiss.index` file.
    ```bash
    python -m backend.ingest
    ```
    *(You should see a message confirming the index build)*

6.  **Run the Backend (FastAPI Server):**
    Open your *first* terminal (ensure `venv` is active) and run:
    ```bash
    uvicorn backend.server:app --reload --port 8000
    ```
    *(Wait for the server to start and, if using a local model, for it to finish loading. Look for `INFO: Application startup complete.`)*

7.  **Run the Frontend (Streamlit App):**
    Open a *second* terminal (ensure `venv` is active) and run:
    ```bash
    streamlit run frontend/app.py
    ```
    *(This will open the application in your web browser)*

---

## Tech Stack

* **Backend:** Python, FastAPI
* **Frontend:** Streamlit
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Google Generative AI (`embedding-001`) or Sentence Transformers (`all-MiniLM-L6-v2`)
* **Generation:** Google Generative AI (`gemini-pro`) or Hugging Face Transformers (`gpt2`, `microsoft/Phi-3-mini-4k-instruct`)
* **Document Processing:** `pdfminer.six`, `beautifulsoup4`
* **Environment Management:** `python-dotenv`

---

## Project Structure

````

├── backend/                \# Contains all FastAPI, RAG logic, and data processing
│   ├── sample\_docs/        \# Sample documents for initial ingestion
│   ├── gemini\_utils.py     \# Functions for interacting with Gemini API
│   ├── ingest.py           \# Document processing and indexing logic
│   ├── rag\_pipeline.py     \# Core RAG prompt formatting (if used)
│   ├── server.py           \# FastAPI application (API endpoints)
│   └── vectorstore.py      \# FAISS index management and embedding logic
│   ├── faiss.index         \# (Generated) FAISS vector index file
│   └── metadata.pkl        \# (Generated) Metadata for indexed chunks
├── frontend/               \# Contains the Streamlit UI code
│   ├── assets/             \# Images, CSS etc. for the frontend
│   │   └── robot.png
│   └── app.py              \# Main Streamlit application script
├── venv/                   \# Python virtual environment (ignored by Git)
├── .env                    \# (Local only\!) API keys and configuration (ignored by Git)
├── .env.example            \# Template for environment variables
├── .gitignore              \# Specifies files/folders for Git to ignore
├── requirements.txt        \# List of Python dependencies
└── README.md               \# This file

````

---

## Known Issues

* **Local Model Memory:** Using large local models (like Phi-3) requires significant RAM (8GB+ recommended). Startup may be slow, and it might fail on low-memory machines. Using `"gpt2"` is faster but provides lower-quality answers.
* **Gemini API Quotas:** The free tier for the Gemini API has rate limits. Heavy usage may require setting up billing on your Google Cloud account.
* **Ingestion Errors:** Document parsing can occasionally fail for complex PDFs or malformed HTML files. Error messages are logged to the backend console.

---

## Contributing

Contributions are welcome! If you'd like to improve ContextIQ, please follow these steps:

1.  **Fork** the repository on GitHub.
2.  Create a new **branch** for your feature or bug fix (`git checkout -b feature/your-feature-name`).
3.  Make your changes and **commit** them (`git commit -m 'Add some amazing feature'`).
4.  **Push** your changes to your fork (`git push origin feature/your-feature-name`).
5.  Create a **Pull Request** back to the main repository.

Please ensure your code follows standard Python conventions and includes relevant updates to documentation if necessary.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file (you should create one) for details.

```text
MIT License

Copyright (c) 2025 Satyam Kumar Pandey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
````

*(Remember to replace `[Year]` and `[Your Name/Organization]` in the license text)*

-----

## Acknowledgements

  * Libraries: FastAPI, Streamlit, FAISS, Hugging Face Transformers, Google Generative AI SDK, Sentence Transformers, pdfminer.six.
  * Inspiration from various RAG tutorials and implementations available online.

<!-- end list -->

```
