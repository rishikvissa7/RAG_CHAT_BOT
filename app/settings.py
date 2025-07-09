from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost"
    QDRANT_PORT: int = 6333
    COLLECTION: str = "documents"
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_MODEL: str = "gemma3"

    class Config:
        env_file = ".env"

settings = Settings()