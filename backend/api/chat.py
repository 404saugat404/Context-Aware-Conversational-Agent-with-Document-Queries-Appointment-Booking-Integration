"""
FastAPI router for chat and appointment endpoints.

Endpoints handle request/response only — all business logic lives in
the services and agents layers.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import ValidationError

from backend.agents.orchestrator import process_message
from backend.config import GEMINI_MODEL_NAME
from backend.logger import get_logger
from backend.models.appointment_schema import (
    AppointmentRequest,
    AppointmentResponse,
    ChatRequest,
    ChatResponse,
    GeminiModel,
)
from backend.services.appointment_service import book_appointment
from backend.rag.ingest import ingest_file

import tempfile
import os

logger = get_logger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.

    Accepts a user message and optional session_id, routes it through
    the LangGraph orchestrator, and returns the agent's reply.
    """
    logger.info(
        "POST /chat — session=%s, message_length=%d",
        request.session_id,
        len(request.message),
    )

    try:
        result = process_message(
            user_message=request.message,
            session_id=request.session_id,
            llm_model=request.model.value if request.model else None,
        )
    except Exception as exc:
        logger.exception("Unhandled error in chat endpoint")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return ChatResponse(
        reply=result["reply"],
        session_id=result["session_id"],
        intent=result.get("intent"),
        sources=result.get("sources"),
        model_used=result.get("model_used"),
    )


@router.post("/appointment", response_model=AppointmentResponse)
async def direct_appointment_endpoint(
    appointment: AppointmentRequest,
) -> AppointmentResponse:
    """
    Direct appointment booking endpoint (bypasses conversational flow).

    Useful for programmatic integrations or testing.
    """
    logger.info(
        "POST /appointment — name=%s, date=%s",
        appointment.name,
        appointment.preferred_date.isoformat(),
    )

    try:
        result = book_appointment(appointment)
    except ValidationError as exc:
        logger.warning("Appointment validation failed: %s", exc)
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Unhandled error in appointment endpoint")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return result


@router.post("/ingest")
async def ingest_document_endpoint(
    file: UploadFile = File(...),
    chunk_size: int = Query(default=512, ge=100, le=2048),
    chunk_overlap: int = Query(default=64, ge=0, le=256),
) -> dict:
    """
    Upload and ingest a document into the vector store.

    Accepts .txt, .md, .csv, or .json files.
    """
    logger.info("POST /ingest — filename=%s, size=%s", file.filename, file.size)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    allowed_extensions = {".txt", ".md", ".csv", ".json"}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}",
        )

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, mode="wb"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        chunks_ingested = ingest_file(
            file_path=tmp_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_metadata={"original_filename": file.filename},
        )
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals():
            os.unlink(tmp_path)

    return {
        "message": f"Successfully ingested '{file.filename}'",
        "chunks_ingested": chunks_ingested,
    }


@router.get("/models")
async def list_models() -> dict:
    """List available Gemini models and the current default."""
    return {
        "available_models": GeminiModel.list_models(),
        "default_model": GEMINI_MODEL_NAME,
    }


@router.get("/health")
async def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "healthy"}
