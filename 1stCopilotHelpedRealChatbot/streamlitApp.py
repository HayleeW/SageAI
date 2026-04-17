import streamlit as st  # type: ignore[import]
try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    from sentence_transformers.util import cos_sim  # type: ignore[import]
except Exception:
    import sys
    st.error(
        "Missing dependency: the 'sentence-transformers' package is not installed or could not be imported.\n\n"
        "Install it with:\n\n"
        "    pip install -U sentence-transformers\n\n"
        "Then restart the Streamlit app."
    )
    st.stop()

import json
import urllib.request
import urllib.error
import os

top_indices = []
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
    return [notes[i] for i in top_indices]

def generate_answer(context, question):
    prompt = f"""Use the following notes to answer the user's question.

    Notes:
    {context}

    Question: {question}

    Explain things clearly, step-by-step, like a patient tutor. Answer in a friendly, conversational tone, like an AI assistant. Don't add anything that isn't supported by the notes, but don't mention the notes. If you don't know the answer, say you don't know, but offer to help with something else.
    """
    try:
        data = json.dumps({
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = resp.read().decode("utf-8")
        parsed = json.loads(resp_body)
        return parsed.get("response", "Sorry, I couldn't generate an answer.")
    except urllib.error.URLError:
        return "Sorry, I couldn't reach the generation server (localhost:11434)."
    except Exception:
        return "Sorry, I couldn't generate an answer."

    
if query.strip():
    
if query.strip():
    with st.spinner("Thinking..."):
        relevant_notes = find_top_notes(query, k=5)
        response = generate_answer(relevant_notes, query)
    st.write(response)
