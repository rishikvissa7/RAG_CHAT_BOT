from typing import List
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# Load the real model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    return model.encode(texts, convert_to_numpy=True).tolist()
