import faiss
from sentence_transformers import SentenceTransformer

# Load the model and index
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("faiss_index.index")

# Sample query
query = "What is the waiting period for pre-existing diseases?"
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n---\n")

# Convert query to embedding
query_vec = model.encode([query], convert_to_numpy=True)

# Search top 3 most similar chunks
top_k = 3
D, I = index.search(query_vec, top_k)

# Show results
print("\nTop matching chunks:")
for i, idx in enumerate(I[0]):
    print(f"\nMatch {i+1} (score: {D[0][i]:.4f}):")
    print(chunks[idx])
