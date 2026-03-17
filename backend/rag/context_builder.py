"""
Assemble the final context string from reranked chunks.

Applies token budgeting so the prompt sent to the LLM stays within limits.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.config import RAG_MAX_CONTEXT_TOKENS
from backend.logger import get_logger

logger = get_logger(__name__)

# Rough chars-per-token ratio (conservative estimate for English text)
_CHARS_PER_TOKEN = 4


def build_context(
    ranked_chunks: List[Dict[str, Any]],
    max_tokens: int = RAG_MAX_CONTEXT_TOKENS,
    separator: str = "\n\n---\n\n",
) -> str:
    """
    Concatenate ranked chunks into a single context string within a token budget.

    Args:
        ranked_chunks: Chunks ordered by relevance (best first).
        max_tokens: Approximate token budget for the assembled context.
        separator: String placed between chunks.

    Returns:
        A combined context string ready to be injected into the LLM prompt.
    """
    if not ranked_chunks:
        logger.warning("build_context received no chunks")
        return ""

    max_chars = max_tokens * _CHARS_PER_TOKEN
    context_parts: list[str] = []
    total_chars = 0

    for chunk in ranked_chunks:
        text = chunk.get("text", "")
        source = chunk.get("metadata", {}).get("source", "unknown")
        entry = f"[Source: {source}]\n{text}"

        if total_chars + len(entry) + len(separator) > max_chars:
            logger.debug(
                "Token budget reached after %d chunks (%d chars)",
                len(context_parts),
                total_chars,
            )
            break

        context_parts.append(entry)
        total_chars += len(entry) + len(separator)

    assembled = separator.join(context_parts)
    logger.info(
        "Built context from %d chunks (~%d tokens)",
        len(context_parts),
        total_chars // _CHARS_PER_TOKEN,
    )
    return assembled
