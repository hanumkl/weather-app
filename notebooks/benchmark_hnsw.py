# Databricks notebook source
# MAGIC %md
# MAGIC # HNSW Index Benchmark
# MAGIC
# MAGIC Stretch goal: compare weather search latency **with** vs **without** the HNSW index.
# MAGIC Runs the same cosine-similarity query N times, drops the index, re-runs, then recreates it.

# COMMAND ----------

# MAGIC %md
# MAGIC Uninstall `psycopg2` / `psycopg2-binary` first — the Databricks runtime already
# MAGIC ships psycopg2, and a pip-installed copy alongside it crashes the kernel.

# COMMAND ----------

# MAGIC %pip uninstall -y psycopg2 psycopg2-binary

# COMMAND ----------

# MAGIC %pip install -q 'databricks-sdk>=0.30.0'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("query", "flash flood risk this weekend", "Test query")
dbutils.widgets.text("runs", "15", "Number of runs per leg")
dbutils.widgets.text("top_k", "5", "Top K")

TEST_QUERY = dbutils.widgets.get("query")
RUNS = int(dbutils.widgets.get("runs"))
TOP_K = int(dbutils.widgets.get("top_k"))

EMBEDDINGS_TABLE = "weather_embeddings"
DOCUMENTS_TABLE = "weather_documents"
INDEX_NAME = f"idx_{EMBEDDINGS_TABLE}_embedding_hnsw"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

# COMMAND ----------

# DBTITLE 1,Connect to Lakebase
import base64
import psycopg2
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
url = base64.b64decode(
    w.secrets.get_secret(scope="database", key="lakebase-url").value
).decode("utf-8")
parsed = urlparse(url)

def get_conn():
    return psycopg2.connect(
        host=parsed.hostname, port=parsed.port or 5432,
        dbname=parsed.path.lstrip("/"),
        user=parsed.username, password=parsed.password,
        sslmode="require",
    )

conn = get_conn()
cur = conn.cursor()
cur.execute(f"SELECT COUNT(*) FROM {EMBEDDINGS_TABLE}")
total = cur.fetchone()[0]
cur.close()
conn.close()
print(f"Rows in {EMBEDDINGS_TABLE}: {total}")
if total == 0:
    dbutils.notebook.exit("weather_embeddings is empty — sync + ingest first.")

# COMMAND ----------

# DBTITLE 1,Embed test query via serving endpoint
vec = w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=[TEST_QUERY]).data[0].embedding
vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]"
print(f"Query embedded to {len(vec)} dims via {EMBEDDING_ENDPOINT}")

# COMMAND ----------

# DBTITLE 1,Benchmark helper
import time
import statistics

def run_bench(label, runs):
    sql = f"""
        SELECT d.id, 1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    conn = get_conn()
    cur = conn.cursor()
    # warm-up
    cur.execute(sql, (vec_str, vec_str, TOP_K))
    _ = cur.fetchall()

    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        cur.execute(sql, (vec_str, vec_str, TOP_K))
        _ = cur.fetchall()
        timings.append((time.perf_counter() - t0) * 1000)

    cur.close()
    conn.close()

    stats = {
        "label": label,
        "runs": runs,
        "mean_ms": round(statistics.mean(timings), 3),
        "median_ms": round(statistics.median(timings), 3),
        "min_ms": round(min(timings), 3),
        "max_ms": round(max(timings), 3),
    }
    print(f"{label}: mean={stats['mean_ms']}ms  median={stats['median_ms']}ms  "
          f"min={stats['min_ms']}ms  max={stats['max_ms']}ms")
    return stats

# COMMAND ----------

# DBTITLE 1,Run: WITH HNSW index
with_stats = run_bench("WITH HNSW", RUNS)

# COMMAND ----------

# DBTITLE 1,Drop index and run: WITHOUT HNSW index
conn = get_conn()
cur = conn.cursor()
cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
conn.commit()
cur.close()
conn.close()
print(f"Dropped {INDEX_NAME}")

without_stats = run_bench("WITHOUT HNSW (seq scan)", RUNS)

# COMMAND ----------

# DBTITLE 1,Recreate index + summary
conn = get_conn()
cur = conn.cursor()
cur.execute(f"""
    CREATE INDEX IF NOT EXISTS {INDEX_NAME}
    ON {EMBEDDINGS_TABLE}
    USING hnsw (embedding vector_cosine_ops)
""")
conn.commit()
cur.close()
conn.close()
print(f"Recreated {INDEX_NAME}")

speedup = without_stats["mean_ms"] / with_stats["mean_ms"] if with_stats["mean_ms"] > 0 else 0
print(f"\n=== Summary ===")
print(f"Query: {TEST_QUERY!r}")
print(f"Rows: {total}")
print(f"Mean with HNSW:    {with_stats['mean_ms']} ms")
print(f"Mean without HNSW: {without_stats['mean_ms']} ms")
print(f"Speedup:           {speedup:.2f}x")
print("Note: HNSW gains show up more on larger tables.")
