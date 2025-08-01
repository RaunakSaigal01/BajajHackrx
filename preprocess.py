import fitz  # PyMuPDF
import re

# Extract full text from the PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# Chunk text based on section or clause-like patterns
def improved_chunking(text):
    pattern = r'(?=(?:\n|^)(?:Section|SECTION|Annexure|ANNEXURE)?\s?\d+(\.\d+)*[.:]?\s[A-Z])'
    raw_chunks = re.split(pattern, text)
    
    chunks = []
    for part in raw_chunks:
        if part and isinstance(part, str):
            cleaned = part.strip().replace("\n", " ")
            if len(cleaned) > 30:
                chunks.append(cleaned)
    return chunks

# Extract, chunk, and print all chunks
def print_chunks(pdf_path):
    text = extract_text(pdf_path)
    chunks = improved_chunking(text)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---\n{chunk}\n")

    print(f"\nTotal chunks: {len(chunks)}")

# Run the function
print_chunks("sample1.pdf")
from sentence_transformers import SentenceTransformer
import numpy as np

# Reuse improved_chunking and extract_text if in a separate file
def embed_chunks(pdf_path, chunk_file="chunks.txt", embedding_file="embeddings.npy"):
    text = extract_text(pdf_path)
    chunks = improved_chunking(text)

    # Load Sentence-BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for each chunk
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    for i in range(3):
        print(f"\nChunk {i+1}:")
        print(chunks[i])
        print("Embedding vector:")
        print(embeddings[i])


   

    # Save embeddings
    np.save(embedding_file, embeddings)

    # Save corresponding chunks
    with open(chunk_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n---\n")

    print(f"Saved {len(chunks)} chunks and corresponding embeddings.")

# Run the embedding step
embed_chunks("sample1.pdf")

