"""
Lakebase (Databricks-managed Postgres) connection helper.

Connects using LAKEBASE_URL resolved from a Databricks secret scope
(scope=database, key=lakebase-url) via the Databricks SDK. The secret
is a base64-encoded Postgres connection URL created by setup_secrets.py.

For local dev, set LAKEBASE_URL in .env as a fallback.
"""

from __future__ import annotations

import base64
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger("weather-app.lakebase")

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

DOCUMENTS_TABLE = os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents")
EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
EMBEDDING_DIM = 384


def _lakebase_url() -> str:
    """
    Resolve the Postgres connection URL.

    Priority:
      1. Databricks secret scope (database/lakebase-url) — production path
      2. LAKEBASE_URL env var — local dev fallback
    """
    # Try Databricks secrets first (the production path)
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        secret = w.secrets.get_secret(scope=_SCOPE, key=_KEY)
        url = base64.b64decode(secret.value).decode("utf-8").strip()
        if url:
            return url
    except Exception:  # noqa: BLE001
        logger.debug("Databricks secret %s/%s not available, trying env var", _SCOPE, _KEY)

    # Fallback: env var (local dev with .env)
    url = os.environ.get("LAKEBASE_URL", "").strip()
    if url:
        return url

    raise RuntimeError(
        f"Cannot connect to Lakebase. Either:\n"
        f"  1. Run setup_secrets.py to store the URL in Databricks scope '{_SCOPE}/{_KEY}', or\n"
        f"  2. Set LAKEBASE_URL in .env for local dev."
    )


@contextmanager
def get_connection() -> Iterator[Any]:
    """Yield a raw psycopg2 connection with a RealDictCursor factory."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE against Lakebase; return affected row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def ensure_pgvector() -> None:
    """Enable the pgvector extension if it is not already present."""
    run_write("CREATE EXTENSION IF NOT EXISTS vector")


def ensure_weather_documents_table() -> None:
    """Create weather_documents (raw NWS narratives) if missing."""
    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
            headline TEXT,
            event TEXT,
            narrative_text TEXT NOT NULL,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_location "
        f"ON {DOCUMENTS_TABLE} (location)"
    )
    run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{DOCUMENTS_TABLE}_source_type "
        f"ON {DOCUMENTS_TABLE} (source_type)"
    )


def ensure_weather_embeddings_table(dim: int = EMBEDDING_DIM) -> None:
    """Create weather_embeddings with a vector(dim) column + HNSW index."""
    ensure_pgvector()
    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES {DOCUMENTS_TABLE}(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector({dim}) NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        )
        """
    )
    run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_document_id "
        f"ON {EMBEDDINGS_TABLE} (document_id)"
    )
    run_write(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_embedding_hnsw
        ON {EMBEDDINGS_TABLE}
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def ensure_schema() -> None:
    """Create both weather tables (and pgvector) if they do not exist."""
    ensure_weather_documents_table()
    ensure_weather_embeddings_table()
    logger.info("Ensured tables %s and %s", DOCUMENTS_TABLE, EMBEDDINGS_TABLE)
