from fastapi import APIRouter
from app.services.rag import RAGPipeline

router = APIRouter()
rag = RAGPipeline()

@router.get("/query/")
def query(q: str):
    result = rag.answer(q)
    return {
        "query": q,
        "answer": result["answer"],
        "mode": result["mode"]
    }