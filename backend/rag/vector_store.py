"""
Qdrant vector store abstraction.

Handles collection creation, upserting document chunks, and search operations.
All Qdrant-specific logic is encapsulated here so the rest of the codebase
interacts only with plain Python types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.config import QDRANT_COLLECTION_NAME, QDRANT_URL
from backend.logger import get_logger

logger = get_logger(__name__)

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Return a shared Qdrant client instance (lazy-initialised)."""
    global _client
    if _client is None:
        logger.info("Connecting to Qdrant at %s", QDRANT_URL)
        _client = QdrantClient(url=QDRANT_URL)
        logger.info("Qdrant client connected")
    return _client


def ensure_collection_exists(
    collection_name: str = QDRANT_COLLECTION_NAME,
    vector_size: int = 384,
) -> None:
    """Create the Qdrant collection if it does not already exist."""
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        logger.debug("Collection '%s' already exists", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logger.info("Created Qdrant collection '%s' (vector_size=%d)", collection_name, vector_size)


def upsert_chunks(
    chunk_texts: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> int:
    """
    Upsert embedded chunks into Qdrant.

    Args:
        chunk_texts: Raw text of each chunk (stored in payload for retrieval).
        embeddings: Corresponding embedding vectors.
        metadatas: Metadata dicts for each chunk.
        collection_name: Target Qdrant collection.

    Returns:
        Number of points upserted.
    """
    if not chunk_texts:
        logger.warning("upsert_chunks called with empty input")
        return 0

    if len(chunk_texts) != len(embeddings) or len(chunk_texts) != len(metadatas):
        raise ValueError(
            f"Length mismatch: texts={len(chunk_texts)}, "
            f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
        )

    client = get_qdrant_client()
    points = [
        PointStruct(
            id=uuid4().hex,
            vector=embedding,
            payload={"text": text, **metadata},
        )
        for text, embedding, metadata in zip(chunk_texts, embeddings, metadatas)
    ]

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Upserted %d points into collection '%s'", len(points), collection_name)
    return len(points)


def search_similar(
    query_embedding: List[float],
    top_k: int = 10,
    collection_name: str = QDRANT_COLLECTION_NAME,
    metadata_filter: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search in Qdrant, with optional metadata filtering.

    Args:
        query_embedding: The query vector.
        top_k: Maximum number of results to return.
        collection_name: Qdrant collection to search.
        metadata_filter: Optional key-value pairs to filter on (AND logic).

    Returns:
        List of dicts with keys ``text``, ``score``, and ``metadata``.
    """
    client = get_qdrant_client()

    qdrant_filter = None
    if metadata_filter:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in metadata_filter.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
        query_filter=qdrant_filter,
    )

    hits: list[dict[str, Any]] = []
    for scored_point in results:
        payload = scored_point.payload or {}
        hits.append(
            {
                "text": payload.get("text", ""),
                "score": scored_point.score,
                "metadata": {k: v for k, v in payload.items() if k != "text"},
            }
        )

    logger.debug(
        "Vector search returned %d results (top_k=%d, filter=%s)",
        len(hits),
        top_k,
        metadata_filter,
    )
    return hits
