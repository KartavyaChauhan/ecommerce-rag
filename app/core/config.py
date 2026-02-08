"""Application configuration using Pydantic Settings with environment variable support."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List


class Settings(BaseSettings):
    """Centralized configuration loaded from environment variables and .env file."""

    PROJECT_NAME: str = "E-Commerce RAG Engine"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Google Gemini API
    GOOGLE_API_KEY: str = Field(..., env="GOOGLE_API_KEY")
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.0-flash"
    LLM_FALLBACK_MODELS: List[str] = ["gemini-2.0-flash-lite", "gemini-2.5-flash"]

    # ChromaDB Vector Store
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    COLLECTION_NAME: str = "ecommerce_docs"

    class Config:
        env_file = ".env"


settings = Settings()