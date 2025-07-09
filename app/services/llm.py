# app/services/llm.py
import requests
from app.settings import settings
from app.logger import logger

class OllamaLLM:
    def __init__(self, model: str = None):
        self.model = model or settings.OLLAMA_MODEL
        self.url = settings.OLLAMA_URL

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        logger.debug("Sending prompt to Ollama (model: %s)", self.model)
        try:
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
                logger.debug("Ollama returned response successfully")
                return response.json()["response"].strip()
            else:
                logger.error("Ollama error %d: %s", response.status_code, response.text)
                return f"[Ollama Error] {response.status_code}: {response.text}"
        except Exception as e:
            logger.exception("Ollama request failed")
            return "[Ollama Error] Request failed"
