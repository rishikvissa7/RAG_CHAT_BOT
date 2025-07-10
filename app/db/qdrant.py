# app/db/qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from app.settings import settings
from app.logger import logger

client = QdrantClient(url=settings.QDRANT_URL, port=settings.QDRANT_PORT)

def init_qdrant_collection(name: str):
    collections = client.get_collections().collections
    if name not in [c.name for c in collections]:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=384, distance="Cosine")
        )

def upsert_documents(docs: list[tuple[str, list[float]]], collection_name: str):
    points = [PointStruct(id=i, vector=vec, payload={"text": txt}) for i, (txt, vec) in enumerate(docs)]
    client.upsert(collection_name=collection_name, points=points)

def query_similar(vec: list[float], collection_name: str, top_k: int = 5):
    hits = client.search(collection_name=collection_name, query_vector=vec, limit=top_k)
    return [hit.payload["text"] for hit in hits]

def get_all_collections():
    return [c.name for c in client.get_collections().collections]

def get_top_match_collection(vec: list[float]):
    collections = get_all_collections()
    max_score = -1
    best_collection = None
    for col in collections:
        hits = client.search(collection_name=col, query_vector=vec, limit=1)
        if hits and hits[0].score > max_score:
            max_score = hits[0].score
            best_collection = col
    return best_collection
