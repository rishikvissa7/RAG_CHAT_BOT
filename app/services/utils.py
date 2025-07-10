# app/services/utils.py
from typing import List
from sentence_transformers import SentenceTransformer
import fitz
from app.db.qdrant import get_top_match_collection
from uuid import uuid4

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def embed_texts(texts: List[str]) -> List[List[float]]:
    return model.encode(texts, convert_to_numpy=True).tolist()

def classify_file(filename, content, private=False):
    if private:
        return f"doc_{uuid4().hex[:8]}"
    elif "resume" in filename or "skills" in content.lower():
        return "resume_docs"
    else:
        return "general_docs"

def get_best_collection(q_vec: list[float]) -> str:
    return get_top_match_collection(q_vec) or "general_docs"
