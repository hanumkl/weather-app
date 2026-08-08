"""
Shared embedding + chunking helpers for weather documents.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) to stay compatible
with the Day 2 news pipeline distance-operator conventions.
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


def get_model():
    """Load the sentence-transformers model once (module-level singleton)."""
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
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    """Embed a single query string (used by the search endpoint)."""
    return embed_texts([query])[0]


def vector_literal(embedding: Sequence[float]) -> str:
    """Format a Python list as a Postgres vector literal: '[v1,v2,...]'."""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


@lru_cache(maxsize=1)
def warm_model() -> str:
    """Eagerly load the model; returns the model name."""
    get_model()
    return EMBEDDING_MODEL_NAME
