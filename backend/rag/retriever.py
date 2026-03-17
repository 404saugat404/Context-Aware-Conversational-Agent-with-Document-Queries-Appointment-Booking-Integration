"""
High-level retriever that wraps vector search and optional metadata filtering.

This is the main entry point for fetching candidate documents given a query string.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.config import RAG_TOP_K
from backend.logger import get_logger
from backend.rag.embedding_model import generate_embeddings
from backend.rag.vector_store import search_similar

logger = get_logger(__name__)


def retrieve_relevant_chunks(
    query: str,
    top_k: int = RAG_TOP_K,
    metadata_filter: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Embed the user query and retrieve the most similar document chunks.

    Args:
        query: The user's natural-language question.
        top_k: Number of candidate chunks to return.
        metadata_filter: Optional metadata constraints (e.g. source file).

    Returns:
        Ranked list of dicts containing ``text``, ``score``, and ``metadata``.
    """
    if not query or not query.strip():
        logger.warning("retrieve_relevant_chunks called with empty query")
        return []

    logger.info("Retrieving top-%d chunks for query: %.80s...", top_k, query)

    query_embedding = generate_embeddings([query])[0]
    results = search_similar(
        query_embedding=query_embedding,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    logger.info("Retrieved %d candidate chunks", len(results))
    return results
