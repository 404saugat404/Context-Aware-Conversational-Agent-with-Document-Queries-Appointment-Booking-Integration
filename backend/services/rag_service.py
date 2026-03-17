"""
RAG service — orchestrates retrieval, reranking, and context assembly.

This is the single entry point that the RAG agent calls. It hides the
multi-step pipeline behind a clean interface.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.logger import get_logger
from backend.rag.context_builder import build_context
from backend.rag.reranker import rerank_chunks
from backend.rag.retriever import retrieve_relevant_chunks
from backend.services.tools_service import rewrite_query_for_retrieval

logger = get_logger(__name__)


def answer_from_documents(
    query: str,
    top_k: int = 10,
    rerank_top_n: int = 3,
    metadata_filter: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve -> rerank -> build context.

    Args:
        query: The user's question.
        top_k: Number of candidates from vector search.
        rerank_top_n: Number of chunks to keep after reranking.
        metadata_filter: Optional metadata constraints.

    Returns:
        Dict with keys ``context`` (assembled text), ``sources`` (list of
        source names), and ``num_chunks_used``.
    """
    logger.info("RAG service handling query: %.100s", query)

    rewritten_query = rewrite_query_for_retrieval(query)
    logger.debug("Using rewritten query for retrieval: '%s'", rewritten_query)

    candidates = retrieve_relevant_chunks(
        query=rewritten_query,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    if not candidates:
        logger.warning("No candidate chunks retrieved for query")
        return {"context": "", "sources": [], "num_chunks_used": 0}

    reranked = rerank_chunks(query=rewritten_query, candidate_chunks=candidates, top_n=rerank_top_n)

    context_text = build_context(reranked)

    sources = list({
        chunk.get("metadata", {}).get("source", "unknown")
        for chunk in reranked
    })

    result = {
        "context": context_text,
        "sources": sources,
        "num_chunks_used": len(reranked),
    }
    logger.info(
        "RAG service complete: %d chunks used, %d sources",
        result["num_chunks_used"],
        len(sources),
    )
    return result
