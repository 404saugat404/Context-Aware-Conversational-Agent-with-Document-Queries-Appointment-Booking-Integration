"""
Embedding and reranker model singletons.

Models are lazily loaded on first access so that import-time cost is zero.
Subsequent calls return the cached instance.

Query embeddings are cached with an LRU cache to avoid redundant computation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

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


@lru_cache(maxsize=128)
def _cached_embed(text: str) -> Tuple[float, ...]:
    """Return a cached embedding for a single text string."""
    model = get_embedding_model()
    vec = model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
    return tuple(vec.tolist())


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate dense vector embeddings for a list of texts.

    Single-text calls (the common query path) hit an LRU cache so repeated
    or similar questions skip model inference entirely.

    Args:
        texts: Plain-text strings to embed.

    Returns:
        A list of embedding vectors (each a list of floats).
    """
    if not texts:
        logger.warning("generate_embeddings called with empty text list")
        return []

    # Fast path: single text (query embedding) → use LRU cache
    if len(texts) == 1:
        logger.debug("Generating embedding for query (cached path)")
        return [list(_cached_embed(texts[0]))]

    # Batch path: multiple texts (e.g. document ingestion) → no cache
    model = get_embedding_model()
    logger.debug("Generating embeddings for %d texts (batch path)", len(texts))
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()
