# app/api/upload.py
from fastapi import APIRouter, UploadFile, File
from typing import List
from app.services.utils import chunk_text, embed_texts, extract_text_from_pdf, classify_file
from app.db.qdrant import upsert_documents, init_qdrant_collection
from app.logger import logger

router = APIRouter()

@router.post("/upload/")
async def upload(files: List[UploadFile] = File(...)):
    logger.info("Received %d file(s) for upload", len(files))

    collection_chunks_map = {}

    for file in files:
        filename = file.filename.lower()
        content = await file.read()
        logger.debug("Processing file: %s", filename)

        try:
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(content)
            elif filename.endswith(".txt"):
                text = content.decode("utf-8")
            else:
                logger.warning("Unsupported file type: %s", filename)
                continue

            collection_name = classify_file(filename, text)

            if collection_name not in collection_chunks_map:
                collection_chunks_map[collection_name] = []

            chunks = chunk_text(text)
            collection_chunks_map[collection_name].extend(chunks)

        except Exception:
            logger.exception("Error processing %s", filename)

    for collection, chunks in collection_chunks_map.items():
        init_qdrant_collection(collection)
        vectors = embed_texts(chunks)
        upsert_documents(list(zip(chunks, vectors)), collection)

    return {"status": "uploaded", "collections": list(collection_chunks_map.keys())}
