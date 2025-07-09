from fastapi import APIRouter, UploadFile, File
from typing import List
from app.services.utils import chunk_text, embed_texts, extract_text_from_pdf
from app.db.qdrant import upsert_documents, clear_collection

router = APIRouter()

@router.post("/upload/")
async def upload(files: List[UploadFile] = File(...)):
    clear_collection()
    all_chunks = []
    for file in files:
        filename = file.filename.lower()
        content = await file.read()
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        elif filename.endswith(".txt"):
            text = content.decode("utf-8")
        else:
            continue
        all_chunks.extend(chunk_text(text))
    vectors = embed_texts(all_chunks)
    upsert_documents(list(zip(all_chunks, vectors)))
    return {"status": "uploaded", "chunks": len(all_chunks)}