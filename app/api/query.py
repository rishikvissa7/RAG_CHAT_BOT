# app/api/query.py
from fastapi import APIRouter
from app.services.rag import RAGPipeline
from app.logger import logger

router = APIRouter()
rag = RAGPipeline()

@router.get("/query/")
def query(q: str):
    logger.info("Received query: %s", q)
    result = rag.answer(q)
    logger.info("Query handled using mode: %s", result["mode"])
    return {
        "query": q,
        "answer": result["answer"],
        "mode": result["mode"]
    }
