import json
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# Load model and FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.index")

# Load the chunks
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n---\n")

# Parameters
THRESHOLD = 1.0  # L2 distance threshold
TOP_K = 3        # top-k results to consider

def find_answer(query):
    """Returns the best-matched clause from chunks for the given query, without leading numbering."""
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_embedding, k=TOP_K)

    min_dist = np.min(D[0])

    if min_dist > THRESHOLD:
        return "Unclear: no relevant information found."

    top_chunk = chunks[I[0][0]].strip()

    # Remove leading numbering like "3.56. " or "1. " etc.
    top_chunk = re.sub(r"^\d+(\.\d+)*\.?\s*", "", top_chunk)

    # Add smart logic based on the content of the top chunk
    lower_chunk = top_chunk.lower()
    if "not covered" in lower_chunk and "until" in lower_chunk:
        return "Yes, covered after waiting period. " + top_chunk
    elif "not covered" in lower_chunk and "excluded" in lower_chunk:
        return "No, permanently excluded. " + top_chunk
    elif "not covered" in lower_chunk:
        return "No, not covered. " + top_chunk
    elif "covered" in lower_chunk or "reimbursed" in lower_chunk or "included" in lower_chunk:
        return "Yes, covered. " + top_chunk
    elif "waiting period" in lower_chunk:
        return "Waiting period details: " + top_chunk
    else:
        return "Unclear: " + top_chunk

if __name__ == "__main__":
    # Simulated API input format
    input_payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf",
        "questions": [
            "What is the grace period for premium payment under the National Policy?",
            "What is the waiting period for pre-existing diseases (PED)?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }

    # Build response
    response = {
        "answers": []
    }

    for q in input_payload["questions"]:
        response["answers"].append(find_answer(q))

    print("\nStructured JSON Response:\n")
    print(json.dumps(response, indent=2))
