from app.services.llm import OllamaLLM
from app.services.utils import embed_texts
from app.db.qdrant import query_similar

class RAGPipeline:
    def __init__(self):
        self.llm = OllamaLLM()

    def answer(self, query: str) -> dict:
        q_vec = embed_texts([query])[0]
        docs = query_similar(q_vec)
        if docs:
            context = "\n\n".join(docs)
            prompt = f"Use the following context to answer the question:\n\n{context}\n\nQ: {query}\nA:"
            response = self.llm.generate(prompt)
            mode = "rag"
        else:
            response = "⚠️ No relevant information found in the uploaded documents."
            mode = "no-context"
        return {"answer": response, "mode": mode}