"""Service for embedding storage and semantic search using ChromaDB."""

import os
from typing import List, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from app.core.config import settings
from app.utils.logger import logger


class VectorService:
    """Manages vector embeddings and similarity search via ChromaDB."""

    def __init__(self):
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        self.vector_db = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embedding_function,
            collection_name=settings.COLLECTION_NAME,
        )
        logger.info(f"ChromaDB initialized at '{settings.CHROMA_PERSIST_DIR}'.")

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """Embed and store text chunks in the vector database.

        Args:
            texts: List of text chunks to embed.
            metadatas: Optional metadata dicts (one per chunk).
        """
        logger.info(f"Adding {len(texts)} chunks to vector database...")
        try:
            self.vector_db.add_texts(texts=texts, metadatas=metadatas)
            logger.info("Chunks persisted successfully.")
        except Exception as e:
            logger.error(f"Failed to add texts to vector DB: {e}")
            raise

    def search_similar(self, query: str, k: int = 4) -> list:
        """Perform semantic similarity search.

        Args:
            query: The search query string.
            k: Number of top results to return.

        Returns:
            List of matching LangChain Document objects.
        """
        logger.info(f"Semantic search: '{query}' (k={k})")
        return self.vector_db.similarity_search(query, k=k)


vector_service = VectorService()