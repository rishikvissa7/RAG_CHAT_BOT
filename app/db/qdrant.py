# app/db/qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from app.settings import settings
from app.logger import logger

client = QdrantClient(url=settings.QDRANT_URL, port=settings.QDRANT_PORT)

def init_qdrant():
    logger.info("Initializing Qdrant collection...")
    collections = client.get_collections().collections
    if settings.COLLECTION not in [c.name for c in collections]:
        logger.info("Creating new collection: %s", settings.COLLECTION)
        client.create_collection(
            collection_name=settings.COLLECTION,
            vectors_config=VectorParams(size=384, distance="Cosine")
        )
    else:
        logger.info("Collection '%s' already exists", settings.COLLECTION)

def upsert_documents(docs: list[tuple[str, list[float]]]):
    logger.info("Upserting %d documents to Qdrant", len(docs))
    points = [PointStruct(id=i, vector=vec, payload={"text": txt}) for i, (txt, vec) in enumerate(docs)]
    client.upsert(collection_name=settings.COLLECTION, points=points)

def query_similar(vec: list[float], top_k: int = 5):
    logger.debug("Querying Qdrant for similar vectors (top_k=%d)", top_k)
    hits = client.search(collection_name=settings.COLLECTION, query_vector=vec, limit=top_k)
    results = [hit.payload["text"] for hit in hits]
    logger.debug("Found %d similar documents", len(results))
    return results

def clear_collection():
    logger.warning("Clearing existing Qdrant collection: %s", settings.COLLECTION)
    client.delete_collection(collection_name=settings.COLLECTION)
    init_qdrant()
