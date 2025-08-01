import faiss
import numpy as np

# Load the embeddings and chunks
embeddings = np.load("embeddings.npy")
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n---\n")

# Create the FAISS index
dimension = embeddings.shape[1]  # should be 384
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance (or use cosine later)

# Add vectors to the index
index.add(embeddings)

# Save the index
faiss.write_index(index, "faiss_index.index")
print(f"Saved FAISS index with {index.ntotal} vectors.")
