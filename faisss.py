import faiss
import numpy as np

def build_faiss_index(chunks, embeddings, index_file=None, save_chunks_file=None):
    """
    Builds a FAISS index from embeddings and returns the index and chunks.

    Parameters:
        chunks (List[str]): Text chunks.
        embeddings (np.ndarray): Corresponding embeddings (shape: n x 384).
        index_file (str, optional): If provided, saves the FAISS index to this path.
        save_chunks_file (str, optional): If provided, saves the chunks to this path.

    Returns:
        index: FAISS index object.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    if index_file:
        faiss.write_index(index, index_file)
        print(f"[✓] FAISS index saved to '{index_file}'")

    if save_chunks_file:
        with open(save_chunks_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n---\n")
        print(f"[✓] Chunks saved to '{save_chunks_file}'")

    return index
