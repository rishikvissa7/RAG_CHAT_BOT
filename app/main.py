from fastapi import FastAPI
from app.db.qdrant import init_qdrant
from app.api import upload, query

app = FastAPI()
init_qdrant()

app.include_router(upload, prefix="/api")
app.include_router(query, prefix="/api")