# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents → Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the **Day 2: Context Engineering on Databricks** homework.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads the `weather_documents` table from Lakebase (synced via `POST /weather/sync`).
# MAGIC 2. Chunks each document's `narrative_text` using a sliding window (800 chars, 100 overlap).
# MAGIC 3. Embeds each chunk with the **`databricks-gte-large-en`** Foundation Model
# MAGIC    endpoint (1024-dim).
# MAGIC 4. Writes embeddings into `weather_embeddings` using psycopg2 + `execute_values`
# MAGIC    with `%s::vector` casts (no Spark JDBC — unreliable against this Lakebase instance).
# MAGIC 5. Creates an HNSW index for fast cosine-similarity search via pgvector's `<=>` operator.
# MAGIC
# MAGIC ### Why a Foundation Model endpoint instead of sentence-transformers?
# MAGIC
# MAGIC The Flask app (`POST /weather/search`) must embed the incoming query with the
# MAGIC **same model** used here, or the cosine scores are meaningless. Databricks Apps
# MAGIC are lightweight containers where torch (~2.5GB) doesn't install reliably, so
# MAGIC both sides call this shared serving endpoint instead. Dimensionality is
# MAGIC therefore **1024**, not MiniLM's 384 — documented in `README_WEATHER.md`.
# MAGIC
# MAGIC **Data source:** National Weather Service API (api.weather.gov) — free, no key needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install packages
# MAGIC
# MAGIC Uninstall `psycopg2` / `psycopg2-binary` first — the Databricks runtime already
# MAGIC ships psycopg2, and having a pip-installed copy alongside it crashes the kernel
# MAGIC ("Fatal error: The Python kernel is unresponsive").
# MAGIC
# MAGIC No `sentence-transformers` / `torch` needed: embeddings come from a serving endpoint.

# COMMAND ----------

# DBTITLE 1,Remove conflicting psycopg2 installs
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary

# COMMAND ----------

# DBTITLE 1,Install SDK
# MAGIC %pip install -q 'databricks-sdk>=0.30.0'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

dbutils.widgets.text("documents_table", "weather_documents", "Source table (raw docs)")
dbutils.widgets.text("embeddings_table", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en", "Embedding endpoint")
dbutils.widgets.text("chunk_size", "800", "Chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Chunk overlap (chars)")
dbutils.widgets.text("request_batch", "32", "Strings per embedding request")
dbutils.widgets.text("max_workers", "1", "Parallel requests (keep 1 on pay-per-token)")
dbutils.widgets.dropdown(
    "allow_parallel", "false", ["false", "true"], "Allow parallel (provisioned throughput only)"
)
dbutils.widgets.text("sleep_between", "1.0", "Seconds to wait between requests")
dbutils.widgets.text("request_timeout", "60", "Per-request timeout (seconds)")
dbutils.widgets.text("max_documents", "0", "Cap documents per run (0 = no cap)")
dbutils.widgets.dropdown("rebuild_all", "false", ["false", "true"], "Re-embed everything")

DOCUMENTS_TABLE = dbutils.widgets.get("documents_table")
EMBEDDINGS_TABLE = dbutils.widgets.get("embeddings_table")
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
REQUEST_BATCH = int(dbutils.widgets.get("request_batch"))
MAX_WORKERS = int(dbutils.widgets.get("max_workers"))
SLEEP_BETWEEN = float(dbutils.widgets.get("sleep_between"))
REQUEST_TIMEOUT = int(dbutils.widgets.get("request_timeout"))
MAX_DOCUMENTS = int(dbutils.widgets.get("max_documents"))
REBUILD_ALL = dbutils.widgets.get("rebuild_all") == "true"
ALLOW_PARALLEL = dbutils.widgets.get("allow_parallel") == "true"

# Widget values persist across runs, so an existing max_workers=8 survives even
# after the default here changes. Parallel requests reliably trip the workspace
# QPS limit on pay-per-token endpoints, so clamp unless explicitly opted in.
if MAX_WORKERS > 1 and not ALLOW_PARALLEL:
    print(
        f"NOTE: max_workers={MAX_WORKERS} would trip the workspace QPS limit on a "
        "pay-per-token endpoint. Forcing 1. Set allow_parallel=true only if this "
        "endpoint has provisioned throughput."
    )
    MAX_WORKERS = 1

# Known output sizes for Databricks Foundation Model embedding endpoints
ENDPOINT_DIMS = {
    "databricks-gte-large-en": 1024,
    "databricks-bge-large-en": 1024,
}
EMBEDDING_DIM = ENDPOINT_DIMS.get(EMBEDDING_ENDPOINT)
if EMBEDDING_DIM is None:
    raise ValueError(
        f"Unknown embedding endpoint {EMBEDDING_ENDPOINT!r} — add its output dimension "
        "to ENDPOINT_DIMS above before running."
    )

# Bump when changing request/retry behaviour — makes stale notebook copies obvious
CODE_VERSION = "2026-08-08-rate-limit-aware"

print(f"Code version: {CODE_VERSION}")
print(f"Endpoint: {EMBEDDING_ENDPOINT} -> {EMBEDDING_DIM}-dim vectors")
print(f"Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
print(f"Tables: {DOCUMENTS_TABLE} -> {EMBEDDINGS_TABLE}")
print(f"Requests: batch={REQUEST_BATCH}, workers={MAX_WORKERS}, "
      f"sleep={SLEEP_BETWEEN}s, timeout={REQUEST_TIMEOUT}s")
print(f"Rebuild all: {REBUILD_ALL}, max_documents: {MAX_DOCUMENTS or 'no cap'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve Lakebase Connection
# MAGIC
# MAGIC Same secret as `lakebase.py` in the Flask app: a base64-encoded Postgres URL
# MAGIC stored in Databricks scope `database`, key `lakebase-url`.

# COMMAND ----------

# DBTITLE 1,Parse Lakebase connection info
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")

lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip("/")
db_user = parsed.username
db_password = parsed.password

print(f"Host: {db_host}:{db_port}")
print(f"Database: {db_name}")
print(f"User: {db_user}")

# COMMAND ----------

# DBTITLE 1,Connection helpers
import psycopg2
from psycopg2.extras import RealDictCursor

def get_conn():
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode="require",
        connect_timeout=10,
    )

def run_ddl(sql, params=None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    cur.close()
    conn.close()

def run_query(sql, params=None):
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

count = run_query(f"SELECT COUNT(*) AS n FROM {DOCUMENTS_TABLE}")[0]["n"]
print(f"Connection successful! Found {count} rows in {DOCUMENTS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure Tables Exist (with dimension migration)
# MAGIC
# MAGIC If `weather_embeddings` already exists with a different `vector(N)` width
# MAGIC (e.g. a previous 384-dim MiniLM run), it is dropped and recreated. Vectors
# MAGIC from different models can't be compared, so old rows must go.

# COMMAND ----------

# DBTITLE 1,Create tables + HNSW index
run_ddl("CREATE EXTENSION IF NOT EXISTS vector")

run_ddl(f"""
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
""")

# atttypmod holds the declared vector width for pgvector columns
existing_dim_rows = run_query(
    """
    SELECT a.atttypmod AS declared_dim
    FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    WHERE c.relname = %s AND a.attname = 'embedding'
    """,
    (EMBEDDINGS_TABLE,),
)

if existing_dim_rows:
    existing_dim = existing_dim_rows[0]["declared_dim"]
    if existing_dim != EMBEDDING_DIM:
        print(
            f"{EMBEDDINGS_TABLE} exists with vector({existing_dim}) but this run produces "
            f"vector({EMBEDDING_DIM}). Dropping and recreating — stale vectors from a "
            f"different model are not comparable."
        )
        run_ddl(f"DROP TABLE {EMBEDDINGS_TABLE}")
    else:
        print(f"{EMBEDDINGS_TABLE} already has the correct vector({EMBEDDING_DIM}) column")

run_ddl(f"""
    CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES {DOCUMENTS_TABLE}(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding vector({EMBEDDING_DIM}) NOT NULL,
        model_name TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (document_id, chunk_index)
    )
""")

run_ddl(f"""
    CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_document_id
    ON {EMBEDDINGS_TABLE} (document_id)
""")

run_ddl(f"""
    CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_embedding_hnsw
    ON {EMBEDDINGS_TABLE}
    USING hnsw (embedding vector_cosine_ops)
""")

print(f"Tables ready: {DOCUMENTS_TABLE}, {EMBEDDINGS_TABLE} (vector({EMBEDDING_DIM}) + HNSW)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Documents to Embed
# MAGIC
# MAGIC By default only documents with no embeddings yet (LEFT JOIN WHERE NULL).
# MAGIC Set the `rebuild_all` widget to `true` to re-embed everything.

# COMMAND ----------

# DBTITLE 1,Query documents needing embeddings
if REBUILD_ALL:
    docs = run_query(f"""
        SELECT d.id, d.narrative_text, d.location, d.source_type, d.headline
        FROM {DOCUMENTS_TABLE} d
        WHERE COALESCE(TRIM(d.narrative_text), '') <> ''
        ORDER BY d.synced_at DESC
    """)
else:
    docs = run_query(f"""
        SELECT d.id, d.narrative_text, d.location, d.source_type, d.headline
        FROM {DOCUMENTS_TABLE} d
        LEFT JOIN {EMBEDDINGS_TABLE} e ON e.document_id = d.id
        WHERE e.document_id IS NULL
          AND COALESCE(TRIM(d.narrative_text), '') <> ''
        ORDER BY d.synced_at DESC
    """)

if MAX_DOCUMENTS > 0 and len(docs) > MAX_DOCUMENTS:
    print(f"Capping this run at {MAX_DOCUMENTS} of {len(docs)} documents "
          f"(re-run to continue with the rest).")
    docs = docs[:MAX_DOCUMENTS]

print(f"Documents to embed: {len(docs)}")
for d in docs[:3]:
    print(f"  [{d['source_type']}] {d['location']}: {d['headline']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Documents
# MAGIC
# MAGIC Sliding-window character chunks: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`.
# MAGIC Most NWS forecast periods are short (1-2 sentences) so they yield a single
# MAGIC chunk; alerts combining `description` + `instruction` may yield 2-3.

# COMMAND ----------

# DBTITLE 1,Build chunks
def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = max(chunk_size - chunk_overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


chunk_rows = []
for doc in docs:
    for idx, text in enumerate(chunk_text(doc["narrative_text"])):
        chunk_rows.append({
            "id": f"{doc['id']}_{idx}",
            "document_id": doc["id"],
            "chunk_index": idx,
            "chunk_text": text,
        })

print(f"Total chunks to embed: {len(chunk_rows)} (from {len(docs)} documents)")
if chunk_rows:
    avg_len = sum(len(c["chunk_text"]) for c in chunk_rows) / len(chunk_rows)
    print(f"  avg chunk length: {avg_len:.0f} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Embeddings
# MAGIC
# MAGIC Calls the endpoint's REST `invocations` API directly rather than through
# MAGIC `serving_endpoints.query()`, which retries internally for up to 5 minutes
# MAGIC with no per-request timeout (that produces `TimeoutError: Timed out after
# MAGIC 0:05:00`).
# MAGIC
# MAGIC ### Rate limits
# MAGIC
# MAGIC Pay-per-token Foundation Model endpoints enforce a **workspace QPS limit**.
# MAGIC Exceeding it returns `429 REQUEST_LIMIT_EXCEEDED`. Because the limit counts
# MAGIC *queries*, not tokens, the strategy is **fewer, larger, spaced-out requests**
# MAGIC rather than parallel ones:
# MAGIC
# MAGIC - `request_batch=32` — many chunks per request, so few requests overall
# MAGIC - `max_workers=1` — sequential; parallelism actively hurts here
# MAGIC - `sleep_between=1.0` — stay under the per-second window
# MAGIC - long exponential backoff with jitter, honouring `Retry-After`
# MAGIC
# MAGIC Raise `max_workers` only if you have a **provisioned throughput** endpoint.

# COMMAND ----------

# DBTITLE 1,Embedding helper (rate-limit aware)
import random
import time

import requests

_host = w.config.host.rstrip("/")
_invocations_url = f"{_host}/serving-endpoints/{EMBEDDING_ENDPOINT}/invocations"


def _post_embeddings(texts, timeout):
    """One HTTP call. Returns (vectors, retry_after_seconds_or_None, error_or_None)."""
    headers = {**w.config.authenticate(), "Content-Type": "application/json"}
    try:
        resp = requests.post(
            _invocations_url, headers=headers, json={"input": texts}, timeout=timeout
        )
    except requests.Timeout:
        return None, None, f"request timed out after {timeout}s"
    except requests.RequestException as exc:
        return None, None, str(exc)

    if resp.status_code == 200:
        data = resp.json().get("data") or []
        if len(data) != len(texts):
            raise RuntimeError(
                f"Requested {len(texts)} embeddings, endpoint returned {len(data)}"
            )
        vectors = []
        for item in data:
            vec = item.get("embedding") or []
            if len(vec) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIM}-dim vectors, got {len(vec)}. "
                    "Update ENDPOINT_DIMS and the vector(N) column."
                )
            vectors.append(vec)
        return vectors, None, None

    if resp.status_code == 429 or resp.status_code >= 500:
        retry_after = resp.headers.get("Retry-After")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else None
        return None, wait, f"HTTP {resp.status_code}: {resp.text[:160]}"

    # Payload too large: caller should split the batch
    if resp.status_code in (400, 413) and len(texts) > 1:
        return None, None, f"SPLIT:HTTP {resp.status_code}: {resp.text[:160]}"

    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")


def embed_batch(texts, timeout=REQUEST_TIMEOUT, max_attempts=6):
    """
    Embed a list of strings, retrying through rate limits.

    Backoff is deliberately long (10s, 20s, 40s, ...) with jitter: the QPS
    window won't clear with 1-2s retries, which is what made the first
    attempt fail on every worker at once.
    """
    last_error = None

    for attempt in range(max_attempts):
        vectors, retry_after, error = _post_embeddings(texts, timeout)
        if vectors is not None:
            return vectors

        # Endpoint rejected the payload size — split and recurse
        if error and error.startswith("SPLIT:"):
            mid = len(texts) // 2
            print(f"    payload rejected, splitting {len(texts)} -> {mid}+{len(texts) - mid}")
            return embed_batch(texts[:mid], timeout) + embed_batch(texts[mid:], timeout)

        last_error = error
        if attempt < max_attempts - 1:
            wait = retry_after if retry_after else 10 * (2 ** attempt)
            wait += random.uniform(0, 2)  # jitter so retries don't align
            print(f"    retry {attempt + 1}/{max_attempts - 1} in {wait:.1f}s ({last_error})")
            time.sleep(wait)

    raise RuntimeError(f"Embedding failed after {max_attempts} attempts: {last_error}")

# COMMAND ----------

# DBTITLE 1,Smoke test one call (confirms endpoint + latency)
t0 = time.perf_counter()
probe = embed_batch(["Sunny with a high near 78."])
print(f"Single-call latency: {(time.perf_counter() - t0) * 1000:.0f} ms")
print(f"Returned {len(probe[0])} dims from {EMBEDDING_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Embed all chunks
if len(chunk_rows) == 0:
    print("No chunks to embed — run POST /weather/sync first!")
    dbutils.notebook.exit("no_data")

texts = [r["chunk_text"] for r in chunk_rows]
batches = [texts[i : i + REQUEST_BATCH] for i in range(0, len(texts), REQUEST_BATCH)]
print(f"Embedding {len(texts)} chunks in {len(batches)} batch(es), "
      f"{MAX_WORKERS} worker(s), {SLEEP_BETWEEN}s between requests...")

t0 = time.perf_counter()

if MAX_WORKERS <= 1:
    all_embeddings = []
    for i, batch in enumerate(batches, start=1):
        all_embeddings.extend(embed_batch(batch))
        print(f"  batch {i}/{len(batches)} -> {len(all_embeddings)}/{len(texts)} chunks")
        if i < len(batches) and SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)
else:
    # Only sensible with provisioned throughput; map preserves input order
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(embed_batch, batches))
    all_embeddings = [vec for batch_vectors in results for vec in batch_vectors]

elapsed = time.perf_counter() - t0

if len(all_embeddings) != len(chunk_rows):
    raise RuntimeError(
        f"Embedding count mismatch: {len(all_embeddings)} vectors for {len(chunk_rows)} chunks"
    )

print(f"Computed {len(all_embeddings)} embeddings ({EMBEDDING_DIM}-dim) in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert Embeddings into Lakebase
# MAGIC
# MAGIC `psycopg2.extras.execute_values` for batch throughput. Each embedding is
# MAGIC cast to the Postgres `vector` type via `%s::vector`. `ON CONFLICT` makes
# MAGIC re-runs idempotent.

# COMMAND ----------

# DBTITLE 1,Write embeddings via psycopg2
from datetime import datetime, timezone
from psycopg2.extras import execute_values

now = datetime.now(timezone.utc).isoformat()

insert_data = [
    (
        row["id"],
        row["document_id"],
        row["chunk_index"],
        row["chunk_text"],
        "[" + ",".join(str(float(x)) for x in vec) + "]",
        EMBEDDING_ENDPOINT,
        now,
    )
    for row, vec in zip(chunk_rows, all_embeddings)
]

print(f"Inserting {len(insert_data)} embeddings into {EMBEDDINGS_TABLE}...")

insert_sql = f"""
    INSERT INTO {EMBEDDINGS_TABLE} (
        id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
    ) VALUES %s
    ON CONFLICT (id) DO UPDATE SET
        chunk_text = EXCLUDED.chunk_text,
        embedding = EXCLUDED.embedding,
        model_name = EXCLUDED.model_name,
        created_at = EXCLUDED.created_at
"""
template = "(%s, %s, %s, %s, %s::vector, %s, %s)"

conn = get_conn()
cur = conn.cursor()
execute_values(cur, insert_sql, insert_data, template=template, page_size=100)
conn.commit()
cur.close()
conn.close()

print(f"Successfully upserted {len(insert_data)} embeddings into {EMBEDDINGS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC Row count plus a live similarity query, so we know retrieval works before
# MAGIC hitting the REST API.

# COMMAND ----------

# DBTITLE 1,Verify with a test similarity search
total = run_query(f"SELECT COUNT(*) AS n FROM {EMBEDDINGS_TABLE}")[0]["n"]
print(f"Total rows in {EMBEDDINGS_TABLE}: {total}")

test_query = "flash flood risk this weekend"
qvec = w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=[test_query]).data[0].embedding
vec_str = "[" + ",".join(str(float(x)) for x in qvec) + "]"

results = run_query(f"""
    SELECT d.location, d.source_type, d.headline, e.chunk_text,
           1 - (e.embedding <=> %s::vector) AS similarity
    FROM {EMBEDDINGS_TABLE} e
    JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
    ORDER BY e.embedding <=> %s::vector
    LIMIT 5
""", (vec_str, vec_str))

print(f"\nTest search: {test_query!r}")
for r in results:
    print(f"  [{r['similarity']:.4f}] ({r['source_type']}) {r['location']}: {r['headline']}")
    print(f"      {r['chunk_text'][:90]}...")

print("\nDone! POST /weather/search can now return results.")
