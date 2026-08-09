"""
Shared embedding + chunking helpers for weather documents.

Embeddings come from **all-MiniLM-L6-v2 (384-dim)**, run locally via `fastembed`
(ONNX Runtime). Rationale:

  - Databricks Free Edition meters model serving as a single per-account pool
    shared by chat and embedding endpoints. Draining it on chat completions
    (the RAG summary) makes the embedding endpoint return 429 as well, which
    takes search down for reasons unrelated to search. Running the model in
    process removes that coupling entirely.
  - `fastembed` is ONNX-based, so it needs no torch (~2.5GB), which does not
    install in a Databricks App container.
  - It is the model the assignment specified.

The ingestion notebook (`notebooks/ingest_weather_embeddings_local.py`) uses the
same weights through `sentence-transformers`. fastembed L2-normalizes its output
and sentence-transformers does not, which does not affect ranking: pgvector's
`<=>` is cosine distance and therefore scale-invariant.

The dimension MUST match the `vector(N)` column in weather_embeddings.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Sequence

logger = logging.getLogger("weather-app.embeddings")

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# Loaded once per process, not per request: the ONNX session is expensive to
# build and completely reusable across queries.
_model = None
_model_lock = threading.Lock()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Sliding-window character chunks (same pattern as the Day 2 news notebook)."""
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


def _get_model():
    """Return the process-wide embedding model, loading it on first use."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Re-check inside the lock: two requests can race past the check above.
        if _model is None:
            from fastembed import TextEmbedding

            logger.info("Loading embedding model %s", EMBEDDING_MODEL)
            _model = TextEmbedding(model_name=EMBEDDING_MODEL)
            logger.info("Embedding model ready")
    return _model


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed strings locally. No network calls, no rate limits."""
    items = list(texts)
    if not items:
        return []

    vectors = [list(map(float, v)) for v in _get_model().embed(items)]

    for vec in vectors:
        if len(vec) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Model {EMBEDDING_MODEL!r} returned {len(vec)}-dim vectors but "
                f"EMBEDDING_DIM is {EMBEDDING_DIM}. Align EMBEDDING_DIM and the "
                f"vector(N) column, then re-run the embedding notebook."
            )
    return vectors


def embed_query(query: str) -> list[float]:
    """Embed a single query string (used by the search endpoint)."""
    return embed_texts([query])[0]


def vector_literal(embedding: Sequence[float]) -> str:
    """Format a Python list as a Postgres vector literal: '[v1,v2,...]'."""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def warm_model() -> str:
    """Load the model ahead of serving traffic so the first search isn't slow."""
    _get_model()
    return EMBEDDING_MODEL


def describe_backend() -> dict:
    """Report the embedding configuration, for /diagnostics."""
    return {
        "backend": "fastembed (ONNX Runtime, in-process)",
        "model": EMBEDDING_MODEL,
        "dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "loaded": _model is not None,
        "note": (
            "The ingestion notebook uses the same MiniLM weights via "
            "sentence-transformers, so query and document vectors share one space. "
            "No Foundation Model API quota is consumed by search."
        ),
    }
