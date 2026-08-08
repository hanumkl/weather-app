# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents → Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the **Day 2: Context Engineering on Databricks** homework.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads the `weather_documents` table from Lakebase (synced via `POST /weather/sync`).
# MAGIC 2. Chunks each document's `narrative_text` using a sliding window (800 chars, 100 overlap).
# MAGIC 3. Embeds each chunk using `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
# MAGIC 4. Writes embeddings into `weather_embeddings` using psycopg2 + `execute_values`
# MAGIC    with `%s::vector` casts (no Spark JDBC — unreliable against this Lakebase instance).
# MAGIC 5. Creates an HNSW index for fast cosine-similarity search via pgvector's `<=>` operator.
# MAGIC
# MAGIC **Data source:** National Weather Service API (api.weather.gov) — free, no key needed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install packages
# MAGIC
# MAGIC Uninstall `psycopg2` / `psycopg2-binary` first — the Databricks runtime already
# MAGIC ships psycopg2, and having a pip-installed copy alongside it crashes the kernel
# MAGIC ("Fatal error: The Python kernel is unresponsive").

# COMMAND ----------

# DBTITLE 1,Remove conflicting psycopg2 installs
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary

# COMMAND ----------

# DBTITLE 1,Install embedding dependencies
# MAGIC %pip install -q sentence-transformers requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

dbutils.widgets.text("documents_table", "weather_documents", "Source table (raw docs)")
dbutils.widgets.text("embeddings_table", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("chunk_size", "800", "Chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Chunk overlap (chars)")
dbutils.widgets.text("batch_size", "32", "Embedding batch size")

DOCUMENTS_TABLE = dbutils.widgets.get("documents_table")
EMBEDDINGS_TABLE = dbutils.widgets.get("embeddings_table")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

EMBEDDING_DIM = 384

print(f"Model: {EMBEDDING_MODEL_NAME} -> {EMBEDDING_DIM}-dim vectors")
print(f"Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
print(f"Tables: {DOCUMENTS_TABLE} -> {EMBEDDINGS_TABLE}")

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

# DBTITLE 1,Test connection
import psycopg2

try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode="require",
        connect_timeout=10,
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {DOCUMENTS_TABLE}")
    count = cursor.fetchone()[0]
    print(f"Connection successful! Found {count} rows in {DOCUMENTS_TABLE}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure Tables Exist
# MAGIC
# MAGIC Creates `weather_documents` and `weather_embeddings` (with pgvector extension)
# MAGIC if they don't already exist.

# COMMAND ----------

# DBTITLE 1,Create tables + HNSW index
def get_conn():
    return psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode="require",
    )

def run_ddl(sql):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()

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

print(f"Tables {DOCUMENTS_TABLE} and {EMBEDDINGS_TABLE} ready (with HNSW index)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch Documents to Embed
# MAGIC
# MAGIC Reads rows from `weather_documents` that do NOT yet have an entry in
# MAGIC `weather_embeddings` (LEFT JOIN WHERE NULL pattern).

# COMMAND ----------

# DBTITLE 1,Query unembedded documents
from psycopg2.extras import RealDictCursor

conn = get_conn()
cur = conn.cursor(cursor_factory=RealDictCursor)
cur.execute(f"""
    SELECT d.id, d.narrative_text, d.location, d.source_type, d.headline
    FROM {DOCUMENTS_TABLE} d
    LEFT JOIN {EMBEDDINGS_TABLE} e ON e.document_id = d.id
    WHERE e.document_id IS NULL
      AND COALESCE(TRIM(d.narrative_text), '') <> ''
    ORDER BY d.synced_at DESC
""")
docs = cur.fetchall()
cur.close()
conn.close()

print(f"Documents to embed: {len(docs)}")
if docs:
    for d in docs[:3]:
        print(f"  [{d['source_type']}] {d['location']}: {d['headline']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk Documents
# MAGIC
# MAGIC Sliding-window character chunks: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`.
# MAGIC Most NWS forecast periods are short (~1-2 sentences), so many documents
# MAGIC produce only 1 chunk. Alerts with combined description + instruction
# MAGIC text may produce 2-3 chunks.

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
    chunks = chunk_text(doc["narrative_text"])
    for idx, text in enumerate(chunks):
        chunk_rows.append({
            "id": f"{doc['id']}_{idx}",
            "document_id": doc["id"],
            "chunk_index": idx,
            "chunk_text": text,
        })

print(f"Total chunks to embed: {len(chunk_rows)}")
print(f"  from {len(docs)} documents")
if chunk_rows:
    avg_len = sum(len(c["chunk_text"]) for c in chunk_rows) / len(chunk_rows)
    print(f"  avg chunk length: {avg_len:.0f} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Embeddings
# MAGIC
# MAGIC Loads `sentence-transformers/all-MiniLM-L6-v2` once and encodes in batches.

# COMMAND ----------

# DBTITLE 1,Embed chunks
from sentence_transformers import SentenceTransformer

if len(chunk_rows) == 0:
    print("No chunks to embed — run POST /weather/sync first!")
    dbutils.notebook.exit("no_data")

print(f"Loading model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Computing embeddings...")
all_embeddings = []
texts = [r["chunk_text"] for r in chunk_rows]
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    vectors = model.encode(batch, show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    print(f"  Embedded {min(i + BATCH_SIZE, len(texts))} / {len(texts)} chunks")

print(f"Computed {len(all_embeddings)} embeddings ({EMBEDDING_DIM}-dim each)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert Embeddings into Lakebase
# MAGIC
# MAGIC Uses `psycopg2.extras.execute_values` for batch insert performance.
# MAGIC Each embedding is cast to Postgres `vector` type via `%s::vector`.
# MAGIC `ON CONFLICT` handles re-runs gracefully.

# COMMAND ----------

# DBTITLE 1,Write embeddings via psycopg2
from datetime import datetime, timezone
from psycopg2.extras import execute_values

now = datetime.now(timezone.utc).isoformat()

insert_data = []
for row, vec in zip(chunk_rows, all_embeddings):
    vec_literal = "[" + ",".join(str(float(x)) for x in vec) + "]"
    insert_data.append((
        row["id"],
        row["document_id"],
        row["chunk_index"],
        row["chunk_text"],
        vec_literal,
        EMBEDDING_MODEL_NAME,
        now,
    ))

if insert_data:
    print(f"Inserting {len(insert_data)} embeddings into {EMBEDDINGS_TABLE}...")

    conn = get_conn()
    cur = conn.cursor()

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

    execute_values(cur, insert_sql, insert_data, template=template, page_size=100)
    inserted_count = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()

    print(f"Successfully upserted {inserted_count} embeddings into {EMBEDDINGS_TABLE}")
else:
    print("No embeddings to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC
# MAGIC Quick sanity check: count rows and run a sample similarity query.

# COMMAND ----------

# DBTITLE 1,Verify embeddings
conn = get_conn()
cur = conn.cursor(cursor_factory=RealDictCursor)

cur.execute(f"SELECT COUNT(*) AS n FROM {EMBEDDINGS_TABLE}")
total = cur.fetchone()["n"]
print(f"Total rows in {EMBEDDINGS_TABLE}: {total}")

cur.execute(f"""
    SELECT e.id, e.chunk_text, d.location, d.source_type, d.headline
    FROM {EMBEDDINGS_TABLE} e
    JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
    LIMIT 5
""")
samples = cur.fetchall()
print(f"\nSample rows:")
for s in samples:
    print(f"  [{s['source_type']}] {s['location']}: {s['headline']}")
    print(f"    chunk: {s['chunk_text'][:100]}...")

# Test a similarity query
if total > 0:
    test_query = "severe weather warning"
    test_vec = model.encode([test_query])[0].tolist()
    vec_str = "[" + ",".join(str(float(x)) for x in test_vec) + "]"
    cur.execute(f"""
        SELECT d.location, d.headline, e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT 3
    """, (vec_str, vec_str))
    results = cur.fetchall()
    print(f"\nTest search: '{test_query}'")
    for r in results:
        print(f"  [{r['similarity']:.4f}] {r['location']}: {r['headline']}")
        print(f"    {r['chunk_text'][:80]}...")

cur.close()
conn.close()
print("\nDone! The POST /weather/search endpoint can now return results.")
