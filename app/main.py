from fastapi import FastAPI
from app.api import upload, query

app = FastAPI()

app.include_router(upload, prefix="/api")
app.include_router(query, prefix="/api")