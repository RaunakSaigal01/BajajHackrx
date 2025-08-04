import re
import numpy as np
from sentence_transformers import SentenceTransformer
from faisss import build_faiss_index           # Your custom FAISS index builder
from preprocess import embed_chunks_from_url  # Your preprocessing function

# Load model only once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Constants
THRESHOLD = 1.0
TOP_K = 3

# Main function to answer question from a given PDF link
def answer_from_pdf_url(pdf_url, question):
    chunks, embeddings = embed_chunks_from_url(pdf_url)
    index = build_faiss_index(chunks, embeddings)

    return find_answer(question, index, chunks)

# Find answer for a question given FAISS index and chunks
def find_answer(question, index, chunks):
    query_embedding = model.encode([question], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_embedding, k=TOP_K)

    if np.min(D[0]) > THRESHOLD:
        return "Unclear: no relevant information found."

    best_chunk = chunks[I[0][0]].strip()
    best_chunk = re.sub(r"^\d+(\.\d+)*\.?\s*", "", best_chunk)
    lower_chunk = best_chunk.lower()

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

# Optional testing from terminal
if __name__ == "__main__":
    url = input("Enter PDF URL: ")
    while True:
        q = input("\nEnter question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        ans = answer_from_pdf_url(url, q)
        print("\nAnswer:\n", ans)
