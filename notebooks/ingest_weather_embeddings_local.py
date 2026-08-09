# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Embeddings — local model (no serving endpoint)
# MAGIC
# MAGIC Fallback path for **Databricks Free Edition**, where Foundation Model API
# MAGIC quota is a per-account pool shared across every pay-per-token endpoint.
# MAGIC Once that pool is spent — e.g. by chat calls for the RAG summary — the
# MAGIC embedding endpoint returns `429 REQUEST_LIMIT_EXCEEDED` too, and no
# MAGIC client-side retry can clear it.
# MAGIC
# MAGIC This notebook runs `sentence-transformers/all-MiniLM-L6-v2` **on the cluster**:
# MAGIC 384-dim, no HTTP calls, no quota, no rate limit. It is also the model the
# MAGIC assignment originally specified.
# MAGIC
# MAGIC Because 384 != 1024, the existing `weather_embeddings` table is dropped and
# MAGIC recreated. Vectors from different models are not comparable, so old rows
# MAGIC must go.

# COMMAND ----------

# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q sentence-transformers 'databricks-sdk>=0.61.0'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Everything in one cell (self-contained)
import base64
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from databricks.sdk import WorkspaceClient
from sentence_transformers import SentenceTransformer

# --- Config ------------------------------------------------------------------
# Leave blank to use the database/lakebase-url secret, exactly like
# ingest_weather_embeddings.py. Whichever path that notebook used, use here —
# writing to a different instance than the app reads is a silent failure.
LAKEBASE_INSTANCE = ""
DOCUMENTS_TABLE = "weather_documents"
EMBEDDINGS_TABLE = "weather_embeddings"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100

w = WorkspaceClient()

if LAKEBASE_INSTANCE:
    _inst = w.database.get_database_instance(name=LAKEBASE_INSTANCE)
    db_host, db_port = _inst.read_write_dns, 5432
    db_name = "databricks_postgres"
    db_user = w.current_user.me().user_name
    db_password = None  # minted per connection below
else:
    _secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    _parsed = urlparse(base64.b64decode(_secret.value).decode("utf-8"))
    db_host, db_port = _parsed.hostname, _parsed.port or 5432
    db_name = _parsed.path.lstrip("/")
    db_user, db_password = _parsed.username, _parsed.password

print(f"Connecting to {db_host}:{db_port}/{db_name} as {db_user}")


def get_conn():
    # Lakebase credentials expire hourly, so mint one per connection.
    password = db_password
    if LAKEBASE_INSTANCE:
        password = w.database.generate_database_credential(
            request_id=str(uuid.uuid4()), instance_names=[LAKEBASE_INSTANCE]
        ).token
    return psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name, user=db_user,
        password=password, sslmode="require", connect_timeout=15,
    )


def run_ddl(sql, params=None):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()
    conn.close()


def run_query(sql, params=None):
    conn = get_conn()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return list(rows)


# --- Migrate the vector column to 384 ----------------------------------------
run_ddl("CREATE EXTENSION IF NOT EXISTS vector")

existing = run_query(
    """
    SELECT a.atttypmod AS declared_dim
    FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    WHERE c.relname = %s AND a.attname = 'embedding'
    """,
    (EMBEDDINGS_TABLE,),
)
if existing and existing[0]["declared_dim"] != EMBEDDING_DIM:
    print(f"Dropping {EMBEDDINGS_TABLE}: vector({existing[0]['declared_dim']}) "
          f"-> vector({EMBEDDING_DIM})")
    run_ddl(f"DROP TABLE {EMBEDDINGS_TABLE}")

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
    CREATE INDEX IF NOT EXISTS idx_{EMBEDDINGS_TABLE}_embedding_hnsw
    ON {EMBEDDINGS_TABLE} USING hnsw (embedding vector_cosine_ops)
""")

# --- Chunk -------------------------------------------------------------------
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


docs = run_query(f"""
    SELECT id, narrative_text, location, source_type, headline
    FROM {DOCUMENTS_TABLE}
    WHERE COALESCE(TRIM(narrative_text), '') <> ''
    ORDER BY synced_at DESC
""")

chunk_rows = [
    {"id": f"{d['id']}_{i}", "document_id": d["id"], "chunk_index": i, "chunk_text": t}
    for d in docs
    for i, t in enumerate(chunk_text(d["narrative_text"]))
]
print(f"{len(docs)} documents -> {len(chunk_rows)} chunks")
assert chunk_rows, "No documents to embed — run POST /weather/sync first"

# --- Embed locally -----------------------------------------------------------
model = SentenceTransformer(MODEL_NAME)
vectors = model.encode(
    [r["chunk_text"] for r in chunk_rows], batch_size=32, show_progress_bar=True
)
print(f"Embedded {len(vectors)} chunks -> {len(vectors[0])} dims")

# --- Upsert ------------------------------------------------------------------
now = datetime.now(timezone.utc).isoformat()
payload = [
    (
        r["id"], r["document_id"], r["chunk_index"], r["chunk_text"],
        "[" + ",".join(str(float(x)) for x in vec) + "]",
        MODEL_NAME, now,
    )
    for r, vec in zip(chunk_rows, vectors)
]

conn = get_conn()
with conn.cursor() as cur:
    execute_values(
        cur,
        f"""
        INSERT INTO {EMBEDDINGS_TABLE} (
            id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            embedding  = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            created_at = EXCLUDED.created_at
        """,
        payload,
        template="(%s, %s, %s, %s, %s::vector, %s, %s)",
        page_size=100,
    )
conn.commit()
conn.close()
print(f"Upserted {len(payload)} embeddings into {EMBEDDINGS_TABLE}")

# --- Verify: live cosine search, embedded locally too ------------------------
test_query = "flash flood risk this weekend"
qvec = model.encode([test_query])[0]
vec_str = "[" + ",".join(str(float(x)) for x in qvec) + "]"

results = run_query(
    f"""
    SELECT d.location, d.source_type, d.headline, e.chunk_text,
           1 - (e.embedding <=> %s::vector) AS similarity
    FROM {EMBEDDINGS_TABLE} e
    JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
    ORDER BY e.embedding <=> %s::vector
    LIMIT 5
    """,
    (vec_str, vec_str),
)

print(f"\nTest search: {test_query!r}")
for r in results:
    print(f"  [{r['similarity']:.4f}] ({r['source_type']}) {r['location']}: {r['headline']}")
    print(f"      {r['chunk_text'][:90]}...")
