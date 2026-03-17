"""
Embedding and reranker model singletons.

Models are lazily loaded on first access so that import-time cost is zero.
Subsequent calls return the cached instance.
"""

from __future__ import annotations

from typing import List, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.config import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME
from backend.logger import get_logger

logger = get_logger(__name__)

_embedding_model: Optional[SentenceTransformer] = None
_reranker_model: Optional[CrossEncoder] = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer embedding model (lazy-loaded)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    """Return the shared CrossEncoder reranker model (lazy-loaded)."""
    global _reranker_model
    if _reranker_model is None:
        logger.info("Loading reranker model: %s", RERANKER_MODEL_NAME)
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        logger.info("Reranker model loaded successfully")
    return _reranker_model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate dense vector embeddings for a list of texts.

    Args:
        texts: Plain-text strings to embed.

    Returns:
        A list of embedding vectors (each a list of floats).
    """
    if not texts:
        logger.warning("generate_embeddings called with empty text list")
        return []

    model = get_embedding_model()
    logger.debug("Generating embeddings for %d texts", len(texts))
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()
