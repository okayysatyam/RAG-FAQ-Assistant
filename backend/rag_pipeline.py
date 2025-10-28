# backend/rag_pipeline.py
from backend.gemini_utils import generate_answer, get_embedding


def rag_query(user_query, retrieved_context):
    """Use RAG logic to combine context and user query"""
    prompt = f"""
    Context: {retrieved_context}
    Question: {user_query}
    Answer in a clear and concise way.
    """
    return generate_answer(prompt)


# Example embedding usage
text = "Artificial Intelligence improves productivity."
print(get_embedding(text))
