from typing import List
import fitz  # PyMuPDF
import numpy as np

def extract_text_from_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def embed_texts(texts: List[str]) -> List[list]:
    # Dummy embedding: replace with your real embedding model
    return [np.random.rand(384).tolist() for _ in texts]