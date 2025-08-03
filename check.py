import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Constants
THRESHOLD = 1.0
TOP_K = 3

# Chunk policy document text using numbered patterns (e.g., 1.1, 2.2, Section 3)
def improved_chunking(text):
    pattern = r'(?=(?:\n|^)(?:Section|SECTION|Annexure|ANNEXURE)?\s?\d+(\.\d+)*[.:]?\s[A-Z])'
    parts = re.split(pattern, text)
    chunks = []
    for part in parts:
        if part and isinstance(part, str):
            cleaned = part.strip().replace("\n", " ")
            if len(cleaned) > 30:
                chunks.append(cleaned)
    return chunks

# Build FAISS index from chunks
def build_faiss_index(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Core logic to find answer for a question from indexed chunks
def find_answer(question, index, chunks):
    query_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_embedding, k=TOP_K)
    min_distance = np.min(D[0])

    if min_distance > THRESHOLD:
        return "Unclear: no relevant information found."

    best_chunk = chunks[I[0][0]].strip()
    best_chunk = re.sub(r"^\d+(\.\d+)*\.?\s*", "", best_chunk)
    lower_chunk = best_chunk.lower()

    # Smart decision logic
    if "not covered" in lower_chunk and "until" in lower_chunk:
        return "Yes, covered after waiting period. " + best_chunk
    elif "not covered" in lower_chunk and "excluded" in lower_chunk:
        return "No, permanently excluded. " + best_chunk
    elif "not covered" in lower_chunk:
        return "No, not covered. " + best_chunk
    elif "covered" in lower_chunk or "reimbursed" in lower_chunk or "included" in lower_chunk:
        return "Yes, covered. " + best_chunk
    elif "waiting period" in lower_chunk:
        return "Waiting period details: " + best_chunk
    else:
        return "Unclear: " + best_chunk
