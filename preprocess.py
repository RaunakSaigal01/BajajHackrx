import re
import fitz  # PyMuPDF
import requests
import numpy as np
from tempfile import NamedTemporaryFile
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def download_pdf(url):
    response = requests.get(url)
    response.raise_for_status()
    tmp_file = NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

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

def embed_chunks_from_url(pdf_url):
    pdf_path = download_pdf(pdf_url)
    text = extract_text(pdf_path)
    chunks = improved_chunking(text)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    return chunks, embeddings
