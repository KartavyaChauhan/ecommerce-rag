"""FastAPI application entry point with CORS, routing, and health check."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.endpoints import router as api_router
from app.utils.logger import logger


def create_application() -> FastAPI:
    """Application factory that creates and configures the FastAPI instance."""

    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="An intelligent RAG-powered Q&A system for E-Commerce product recommendations.",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router, prefix=settings.API_V1_STR)

    @application.on_event("startup")
    async def startup_event():
        logger.info(f"{settings.PROJECT_NAME} v{settings.VERSION} starting up...")

    @application.get("/health", tags=["Health"])
    async def health_check():
        """Returns application health status."""
        return {"status": "healthy", "version": settings.VERSION}

    return application


app = create_application()