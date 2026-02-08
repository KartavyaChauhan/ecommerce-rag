"""RAG service: retrieves context from the vector store and generates answers via Gemini."""

from typing import Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.services.vector_service import vector_service
from app.utils.logger import logger

SYSTEM_PROMPT = """\
You are a helpful and professional AI assistant for an E-Commerce system.

Answer the question based ONLY on the following context.
If the answer is not in the context, strictly say "I don't have enough information to answer that."
Do not make up facts.

Context:
{context}

Question:
{question}
"""


class RAGService:
    """Orchestrates retrieval-augmented generation with automatic LLM fallback."""

    def __init__(self):
        self.model_names: List[str] = [settings.LLM_MODEL] + settings.LLM_FALLBACK_MODELS

        self.llms = [
            ChatGoogleGenerativeAI(
                model=name,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.2,
                convert_system_message_to_human=True,
            )
            for name in self.model_names
        ]
        logger.info(f"RAG Service ready — models: {self.model_names}")

        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

    @staticmethod
    def _format_docs(docs) -> str:
        """Concatenate document page contents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _deduplicate(docs) -> list:
        """Remove duplicate chunks based on content."""
        seen, unique = set(), []
        for doc in docs:
            if doc.page_content not in seen:
                unique.append(doc)
                seen.add(doc.page_content)
        return unique

    async def generate_answer(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Execute the full RAG pipeline: retrieve → deduplicate → generate → respond.

        Args:
            query: The user's natural-language question.
            k: Number of context chunks to retrieve.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of dicts).
        """
        logger.info(f"RAG query: '{query}'")

        docs = vector_service.search_similar(query, k)
        if not docs:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
            }

        unique_docs = self._deduplicate(docs)
        context_text = self._format_docs(unique_docs)

        last_error = None
        for llm, model_name in zip(self.llms, self.model_names):
            try:
                logger.info(f"Invoking model: {model_name}")
                chain = self.prompt | llm | StrOutputParser()
                answer = await chain.ainvoke({"context": context_text, "question": query})

                logger.info(f"Answer generated successfully via {model_name}.")
                sources = [
                    {
                        "source": doc.metadata.get("source", "unknown"),
                        "content": doc.page_content[:200] + "...",
                    }
                    for doc in unique_docs
                ]
                return {"answer": answer, "sources": sources}

            except Exception as e:
                last_error = e
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    logger.warning(f"{model_name} rate-limited — falling back...")
                    continue
                logger.error(f"LLM error on {model_name}: {e}")
                raise

        logger.error(f"All models exhausted. Last error: {last_error}")
        raise last_error  # type: ignore[misc]


rag_service = RAGService()
