import requests
from app.settings import settings

class OllamaLLM:
    def __init__(self, model: str = None):
        self.model = model or settings.OLLAMA_MODEL
        self.url = settings.OLLAMA_URL

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        response = requests.post(self.url, json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_tokens
            }
        })
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"[Ollama Error] {response.status_code}: {response.text}"