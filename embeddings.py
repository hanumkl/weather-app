"""
Shared embedding + chunking helpers for weather documents.

Two embedding backends:
  1. sentence-transformers (local) — used by the notebook on a cluster
  2. Databricks Foundation Model API — used by the Flask app (no torch needed)

Both produce 384-dim vectors compatible with the same pgvector <=> queries.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Sequence

logger = logging.getLogger("weather-app.embeddings")

EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

_model = None
_USE_LOCAL = None


def _can_use_local() -> bool:
    """Check if sentence-transformers + torch are available (e.g. on a cluster)."""
    global _USE_LOCAL
    if _USE_LOCAL is not None:
        return _USE_LOCAL
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        _USE_LOCAL = True
    except ImportError:
        _USE_LOCAL = False
        logger.info(
            "sentence-transformers not installed — will use Databricks Foundation Model API "
            "for query embedding. This is normal for Databricks App deployments."
        )
    return _USE_LOCAL


def get_model():
    """Load the sentence-transformers model once (only works on clusters with torch)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s ...", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model ready (%s-dim)", EMBEDDING_DIM)
    return _model


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Sliding-window character chunks (same pattern as Day 2 news notebook)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def embed_texts(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    """Encode a batch of strings into 384-dim float vectors."""
    if not texts:
        return []

    if _can_use_local():
        model = get_model()
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return [v.tolist() for v in vectors]

    return _embed_via_foundation_model(list(texts))


def _embed_via_foundation_model(texts: list[str]) -> list[list[float]]:
    """
    Fallback: use Databricks Foundation Model Serving for embeddings.
    Works inside Databricks Apps without needing torch installed.
    """
    from databricks.sdk import WorkspaceClient

    endpoint = os.environ.get(
        "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-bge-large-en"
    )
    w = WorkspaceClient()
    all_vectors: list[list[float]] = []

    for text in texts:
        response = w.serving_endpoints.query(
            name=endpoint,
            input=text,
        )
        if hasattr(response, "data") and response.data:
            vec = response.data[0].embedding
            # Pad or truncate to 384-dim if the Foundation Model returns a different size
            if len(vec) > EMBEDDING_DIM:
                vec = vec[:EMBEDDING_DIM]
            elif len(vec) < EMBEDDING_DIM:
                vec = vec + [0.0] * (EMBEDDING_DIM - len(vec))
            all_vectors.append(vec)
        else:
            raise RuntimeError(
                f"Databricks embedding endpoint '{endpoint}' returned no data. "
                "Check that the endpoint exists and is running."
            )

    return all_vectors


def embed_query(query: str) -> list[float]:
    """Embed a single query string (used by the search endpoint)."""
    return embed_texts([query])[0]


def vector_literal(embedding: Sequence[float]) -> str:
    """Format a Python list as a Postgres vector literal: '[v1,v2,...]'."""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


@lru_cache(maxsize=1)
def warm_model() -> str:
    """Eagerly load the model if available; returns the model name."""
    if _can_use_local():
        get_model()
    return EMBEDDING_MODEL_NAME
