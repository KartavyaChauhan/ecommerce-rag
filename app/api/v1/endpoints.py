"""API v1 endpoints for document upload, semantic search, and RAG chat."""

from fastapi import APIRouter, UploadFile, File, HTTPException, Body

from app.services.document_service import document_service
from app.services.vector_service import vector_service
from app.services.rag_service import rag_service
from app.models.schemas import DocumentResponse, ChatRequest, ChatResponse

router = APIRouter(tags=["RAG"])


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document, extract text, chunk it, and store embeddings."""
    try:
        chunks = await document_service.process_file(file)
        metadatas = [{"source": file.filename} for _ in chunks]
        vector_service.add_texts(texts=chunks, metadatas=metadatas)

        return DocumentResponse(
            filename=file.filename,
            content_type=file.content_type,
            size=file.size or 0,
            chunks_created=len(chunks),
            message="File processed and embeddings stored successfully.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(query: str = Body(..., embed=True), k: int = 3):
    """Perform a raw semantic search and return matching text chunks."""
    try:
        results = vector_service.search_similar(query, k=k)
        return {
            "query": query,
            "results": [doc.page_content for doc in results],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_with_docs(request: ChatRequest):
    """Ask a question — the system retrieves relevant context and generates an answer."""
    try:
        response = await rag_service.generate_answer(request.query, request.k)
        return ChatResponse(
            answer=response["answer"],
            sources=response["sources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))