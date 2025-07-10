# app/services/rag.py
from app.services.llm import OllamaLLM
from app.services.utils import embed_texts, get_best_collection
from app.db.qdrant import query_similar
from app.logger import logger

class RAGPipeline:
    def __init__(self):
        self.llm = OllamaLLM()

    def answer(self, query: str) -> dict:
        logger.debug("Embedding query")
        q_vec = embed_texts([query])[0]

        logger.debug("Selecting best matching collection")
        collection = get_best_collection(q_vec)

        logger.info("Querying collection: %s", collection)
        docs = query_similar(q_vec, collection)

        if docs:
            context = "\n\n".join(docs)
            prompt = f"Use the following context to answer the question:\n\n{context}\n\nQ: {query}\nA:"
            response = self.llm.generate(prompt)
            mode = "rag"
        else:
            response = "⚠️ No relevant information found in the uploaded documents."
            mode = "no-context"

        return {"answer": response, "mode": mode}
