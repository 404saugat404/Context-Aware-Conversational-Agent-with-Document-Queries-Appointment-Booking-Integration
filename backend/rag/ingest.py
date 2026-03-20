"""
Document ingestion pipeline.

Reads files from disk, chunks them, embeds the chunks, and upserts
into Qdrant. This module is meant to be run as a one-off or scheduled job.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from backend.logger import get_logger
from backend.rag.chunking import DocumentChunk, chunk_document
from backend.rag.embedding_model import generate_embeddings
from backend.rag.vector_store import ensure_collection_exists, upsert_chunks

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}


def read_file(file_path: str) -> str:
    """Read an entire file and return its contents as a string."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )
    content = path.read_text(encoding="utf-8")
    logger.debug("Read %d characters from %s", len(content), file_path)
    return content


def ingest_file(
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    extra_metadata: Optional[dict] = None,
) -> int:
    """
    Ingest a single file into the vector store.

    Steps:
        1. Read the file.
        2. Chunk the text.
        3. Generate embeddings.
        4. Upsert into Qdrant.

    Args:
        file_path: Path to the document file.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        extra_metadata: Additional metadata to attach to every chunk.

    Returns:
        Number of chunks ingested.
    """
    logger.info("Starting ingestion for file: %s", file_path)
    ensure_collection_exists()

    text = read_file(file_path)

    # Prefer the original filename from extra_metadata over the temp file name
    if extra_metadata and extra_metadata.get("original_filename"):
        source_name = extra_metadata["original_filename"]
    else:
        source_name = os.path.basename(file_path)

    chunks: List[DocumentChunk] = chunk_document(
        text=text,
        source=source_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extra_metadata=extra_metadata,
    )

    if not chunks:
        logger.warning("No chunks produced from %s — skipping", file_path)
        return 0

    chunk_texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    embeddings = generate_embeddings(chunk_texts)
    upserted = upsert_chunks(chunk_texts, embeddings, metadatas)

    logger.info("Ingested %d chunks from %s", upserted, file_path)
    return upserted


def ingest_directory(
    directory_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    """
    Ingest all supported files in a directory (non-recursive).

    Returns:
        Total number of chunks ingested across all files.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    total = 0
    files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        logger.warning("No supported files found in %s", directory_path)
        return 0

    logger.info("Found %d files to ingest in %s", len(files), directory_path)
    for file_path in files:
        try:
            count = ingest_file(str(file_path), chunk_size, chunk_overlap)
            total += count
        except Exception:
            logger.exception("Failed to ingest %s — skipping", file_path)

    logger.info("Directory ingestion complete: %d total chunks from %s", total, directory_path)
    return total
