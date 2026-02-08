"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel, Field
from typing import List


class DocumentResponse(BaseModel):
    """Response returned after a document is uploaded and processed."""
    filename: str
    content_type: str
    size: int
    chunks_created: int
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str


class ChatRequest(BaseModel):
    """Payload for the /chat endpoint."""
    query: str = Field(..., min_length=1, description="The user's question.")
    k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve.")


class Source(BaseModel):
    """A single source chunk returned alongside an answer."""
    source: str
    content: str


class ChatResponse(BaseModel):
    """Response containing the LLM-generated answer and supporting sources."""
    answer: str
    sources: List[Source]