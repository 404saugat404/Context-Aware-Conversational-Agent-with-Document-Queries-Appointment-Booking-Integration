"""
Document chunking strategies.

Provides recursive text splitting with configurable chunk size and overlap.
Each chunk carries metadata (source, chunk_id, section) for downstream filtering.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from backend.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class DocumentChunk:
    """A single chunk of text with associated metadata."""

    text: str
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = ""
    section: str = ""
    metadata: dict = field(default_factory=dict)


def _recursive_split(
    text: str,
    max_length: int,
    overlap: int,
    separators: list[str],
) -> list[str]:
    """Split *text* recursively using the first effective separator."""
    if len(text) <= max_length:
        return [text]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        chunks: list[str] = []
        current = parts[0]
        for part in parts[1:]:
            candidate = current + sep + part
            if len(candidate) <= max_length:
                current = candidate
            else:
                chunks.append(current)
                # apply overlap by keeping tail of previous chunk
                overlap_text = current[-overlap:] if overlap else ""
                current = overlap_text + part
        if current:
            chunks.append(current)
        return chunks

    # Fallback: hard split by character limit
    return [text[i : i + max_length] for i in range(0, len(text), max_length - overlap)]


def chunk_document(
    text: str,
    source: str = "",
    section: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    extra_metadata: Optional[dict] = None,
) -> List[DocumentChunk]:
    """
    Split a document into overlapping chunks with metadata.

    Args:
        text: The raw document text.
        source: Identifier for the document origin (e.g. filename).
        section: Optional section/heading label.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.
        extra_metadata: Any additional key-value pairs to attach.

    Returns:
        A list of ``DocumentChunk`` objects ready for embedding.
    """
    if not text or not text.strip():
        logger.warning("chunk_document called with empty text (source=%s)", source)
        return []

    raw_chunks = _recursive_split(text, chunk_size, chunk_overlap, SEPARATORS)
    # Filter out whitespace-only chunks
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

    # Prefer original_filename over temp file path for the source field
    display_source = (extra_metadata or {}).get("original_filename", source)

    doc_chunks: list[DocumentChunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        metadata = {
            "source": display_source,
            "section": section,
            "chunk_index": idx,
            **(extra_metadata or {}),
        }
        doc_chunks.append(
            DocumentChunk(
                text=chunk_text,
                source=display_source,
                section=section,
                metadata=metadata,
            )
        )

    logger.info(
        "Chunked document source=%s into %d chunks (chunk_size=%d, overlap=%d)",
        source,
        len(doc_chunks),
        chunk_size,
        chunk_overlap,
    )
    return doc_chunks
