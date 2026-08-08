"""
Shared embedding + chunking helpers for weather documents.

Embeddings come from a Databricks Foundation Model serving endpoint
(`databricks-gte-large-en`, 1024-dim) rather than a local
sentence-transformers model. Rationale:

  - Databricks Apps are lightweight containers; torch (~2.5GB) does not
    install reliably there.
  - Using one endpoint for BOTH ingestion (notebook) and query embedding
    (Flask app) guarantees the two sides share a vector space. Mixing models
    silently produces meaningless cosine scores.

The dimension MUST match the `vector(N)` column in weather_embeddings.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

logger = logging.getLogger("weather-app.embeddings")

EMBEDDING_ENDPOINT = os.environ.get(
    "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))

# Max strings per serving-endpoint request
_REQUEST_BATCH = int(os.environ.get("EMBED_REQUEST_BATCH", "16"))
# Per-request timeout so a slow endpoint can't hang a web request
REQUEST_TIMEOUT = int(os.environ.get("EMBED_REQUEST_TIMEOUT", "30"))


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


def embed_texts(texts: Sequence[str], batch_size: int = _REQUEST_BATCH) -> list[list[float]]:
    """
    Embed strings via the Databricks Foundation Model endpoint.

    Calls the REST `invocations` API rather than `serving_endpoints.query()`,
    which retries internally for up to 5 minutes with no per-request timeout —
    long enough to hang an HTTP request handler.
    """
    items = list(texts)
    if not items:
        return []

    import requests
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    url = f"{w.config.host.rstrip('/')}/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations"
    headers = {**w.config.authenticate(), "Content-Type": "application/json"}

    vectors: list[list[float]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        resp = requests.post(
            url, headers=headers, json={"input": batch}, timeout=REQUEST_TIMEOUT
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding endpoint {EMBEDDING_ENDPOINT!r} returned "
                f"HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json().get("data") or []
        if not data:
            raise RuntimeError(
                f"Embedding endpoint {EMBEDDING_ENDPOINT!r} returned no data. "
                "Check that it exists and this identity can query it."
            )

        for item in data:
            vec = list(item.get("embedding") or [])
            if len(vec) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Endpoint {EMBEDDING_ENDPOINT!r} returned {len(vec)}-dim vectors "
                    f"but EMBEDDING_DIM is {EMBEDDING_DIM}. Update EMBEDDING_DIM and the "
                    f"vector(N) column so they agree, then re-run the embedding notebook."
                )
            vectors.append(vec)

    return vectors


def embed_query(query: str) -> list[float]:
    """Embed a single query string (used by the search endpoint)."""
    return embed_texts([query])[0]


def vector_literal(embedding: Sequence[float]) -> str:
    """Format a Python list as a Postgres vector literal: '[v1,v2,...]'."""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def warm_model() -> str:
    """
    No local model to load — kept so callers have a single place to verify the
    endpoint is reachable before serving traffic. Returns the endpoint name.
    """
    return EMBEDDING_ENDPOINT


def describe_backend() -> dict:
    """Report the embedding configuration, for /diagnostics."""
    return {
        "backend": "databricks foundation model",
        "endpoint": EMBEDDING_ENDPOINT,
        "dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "note": (
            "The embedding notebook uses this same endpoint, so query and document "
            "vectors share one space."
        ),
    }
