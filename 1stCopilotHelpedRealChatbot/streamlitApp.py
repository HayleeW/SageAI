import streamlit as st
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import requests
import os

top_indicies = []
k = 5
notes = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_DIR = os.path.join(BASE_DIR, "Notes")

for filename in os.listdir(NOTES_DIR):
    filepath = os.path.join(NOTES_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                notes.append(line)

st.title("SageAI")

# Ask user for a query
query = st.text_input("Ask me something...", value="", key="query_input")

model = SentenceTransformer("all-MiniLM-L6-v2")
note_embeddings = model.encode(notes)


def find_top_notes(query, k=5):
    query_embedding = model.encode([query])[0]

    # Compute similarity scores
    scores = cos_sim(query_embedding, note_embeddings)[0]
    # Find the index of the most similar note
    top_indices = scores.topk(k).indices.tolist()
    # Return the best matching note
    return [notes[i] for i in top_indices]

def generate_answer(context, question):
    prompt = f"""Use the following notes to answer the user's question.

    Notes:
    {context}

    Question: {question}

    Explain things clearly, step-by-step, like a patient tutor. Answer in a friendly, conversational tone, like an AI assistant. Don't add anything that isn't supported by the notes, but don't mention the notes. If you don't know the answer, say you don't know, but offer to help with something else.
    """
    response = requests.post(
    "http://localhost:11434/api/generate",
    json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
)
    return response.json().get("response", "Sorry, I couldn't generate an answer.")

    
if query.strip():
    with st.spinner("Thinking..."):
        relevant_notes = find_top_notes(query, k=5)
        response = generate_answer(relevant_notes, query)
    st.write(response)