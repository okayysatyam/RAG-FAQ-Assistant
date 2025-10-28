from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ---------- Text Generation ----------
def generate_answer(prompt):
    """Generates text response from Gemini Pro model"""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text


# ---------- Embeddings ----------
def get_embedding(text):
    """Generates vector embedding for given text"""
    embed_model = "models/embedding-001"
    embedding = genai.embed_content(model=embed_model, content=text)
    return embedding["embedding"]
