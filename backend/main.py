"""
FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Context-Aware Conversational Agent",
    description=(
        "RAG-powered document Q&A with conversational appointment booking. "
        "Uses LangGraph for intent routing, Qdrant for vector search, "
        "and Gemini for LLM generation."
    ),
    version="1.0.0",
)

# CORS — allow all origins in development; lock down for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1", tags=["chat"])


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Application starting up")

    # Pre-load ML models so the first request isn't penalised
    from backend.rag.embedding_model import get_embedding_model, get_reranker_model

    logger.info("Pre-loading embedding model...")
    get_embedding_model()
    logger.info("Pre-loading reranker model...")
    get_reranker_model()
    logger.info("All models loaded")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Application shutting down")
