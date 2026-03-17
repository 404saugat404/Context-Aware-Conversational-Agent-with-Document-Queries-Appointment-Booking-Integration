"""
Cross-encoder reranker that refines the initial retriever results.

Takes a query and candidate chunks, scores each pair, and returns the
top-N most relevant chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.config import RAG_RERANK_TOP_N
from backend.logger import get_logger
from backend.rag.embedding_model import get_reranker_model

logger = get_logger(__name__)


def rerank_chunks(
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    top_n: int = RAG_RERANK_TOP_N,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks using a cross-encoder model.

    Args:
        query: The user's question.
        candidate_chunks: Output from the retriever, each with at least a ``text`` key.
        top_n: How many top results to keep after reranking.

    Returns:
        The *top_n* most relevant chunks, sorted by reranker score descending.
    """
    if not candidate_chunks:
        logger.warning("rerank_chunks called with no candidates")
        return []

    reranker = get_reranker_model()
    pairs = [(query, chunk["text"]) for chunk in candidate_chunks]

    logger.debug("Reranking %d candidates", len(pairs))
    scores = reranker.predict(pairs)

    # Attach reranker score and sort descending
    for chunk, score in zip(candidate_chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(candidate_chunks, key=lambda c: c["rerank_score"], reverse=True)
    top_results = ranked[:top_n]

    logger.info(
        "Reranked %d -> %d chunks (top rerank_score=%.4f)",
        len(candidate_chunks),
        len(top_results),
        top_results[0]["rerank_score"] if top_results else 0.0,
    )
    return top_results
