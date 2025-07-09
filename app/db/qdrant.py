from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from app.settings import settings

client = QdrantClient(url=settings.QDRANT_URL, port=settings.QDRANT_PORT)

def init_qdrant():
    collections = client.get_collections().collections
    if settings.COLLECTION not in [c.name for c in collections]:
        client.create_collection(
            collection_name=settings.COLLECTION,
            vectors_config=VectorParams(size=384, distance="Cosine")
        )

def upsert_documents(docs: list[tuple[str, list[float]]]):
    points = [PointStruct(id=i, vector=vec, payload={"text": txt}) for i, (txt, vec) in enumerate(docs)]
    client.upsert(collection_name=settings.COLLECTION, points=points)

def query_similar(vec: list[float], top_k: int = 5):
    hits = client.search(collection_name=settings.COLLECTION, query_vector=vec, limit=top_k)
    return [hit.payload["text"] for hit in hits]

def clear_collection():
    client.delete_collection(collection_name=settings.COLLECTION)
    init_qdrant()