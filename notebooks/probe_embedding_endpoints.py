# Databricks notebook source
# MAGIC %md
# MAGIC # Probe Embedding Endpoints
# MAGIC
# MAGIC Run this when `ingest_weather_embeddings` keeps failing with
# MAGIC `429 REQUEST_LIMIT_EXCEEDED`. Pay-per-token Foundation Model endpoints share
# MAGIC one workspace-wide request budget, so a busy workspace can leave a given
# MAGIC endpoint unusable no matter how the client retries.
# MAGIC
# MAGIC This notebook sends **one small request** to each embedding endpoint you can
# MAGIC see and reports which ones respond, how fast, and how many dimensions they
# MAGIC return. Use the result to pick a working endpoint.
# MAGIC
# MAGIC It deliberately does not retry: the point is a fast snapshot of what is
# MAGIC reachable right now.

# COMMAND ----------

# MAGIC %pip install -q 'databricks-sdk>=0.30.0'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,List every serving endpoint this identity can see
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

endpoints = list(w.serving_endpoints.list())
print(f"{len(endpoints)} serving endpoint(s) visible:\n")
for e in endpoints:
    state = getattr(e.state, "ready", None) if e.state else None
    print(f"  {e.name:<45} ready={state}")

# COMMAND ----------

# DBTITLE 1,Test each candidate embedding endpoint
import time

import requests

# Anything that looks like an embedding model, plus known Databricks defaults
KNOWN = ["databricks-gte-large-en", "databricks-bge-large-en"]
candidates = sorted(
    {e.name for e in endpoints if any(k in e.name.lower() for k in ("embed", "gte", "bge"))}
    | set(KNOWN)
)

host = w.config.host.rstrip("/")
headers = {**w.config.authenticate(), "Content-Type": "application/json"}

print(f"Testing {len(candidates)} candidate(s) with a single short request each.\n")

working = []
for name in candidates:
    url = f"{host}/serving-endpoints/{name}/invocations"
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url, headers=headers, json={"input": ["flash flood warning"]}, timeout=30
        )
    except Exception as exc:  # noqa: BLE001 - report anything that goes wrong
        print(f"  {name:<45} ERROR    {type(exc).__name__}: {exc}")
        continue

    ms = (time.perf_counter() - t0) * 1000

    if resp.status_code == 200:
        dims = len((resp.json().get("data") or [{}])[0].get("embedding") or [])
        print(f"  {name:<45} OK       {dims}-dim, {ms:.0f} ms")
        working.append((name, dims))
    elif resp.status_code == 429:
        print(f"  {name:<45} THROTTLED  budget exhausted right now")
    elif resp.status_code in (403, 404):
        print(f"  {name:<45} NO ACCESS  HTTP {resp.status_code}")
    else:
        print(f"  {name:<45} HTTP {resp.status_code}: {resp.text[:90]}")

# COMMAND ----------

# DBTITLE 1,What to do with the result
if not working:
    print(
        "No embedding endpoint is usable right now.\n\n"
        "Every candidate is throttled or inaccessible, which is a workspace "
        "capacity problem rather than anything in this repo. Options:\n"
        "  1. Wait and re-run — shared budgets free up as others stop.\n"
        "  2. Ask whoever runs the workspace for a provisioned throughput endpoint.\n"
        "Your already-embedded chunks are safe; the ingest notebook resumes."
    )
else:
    print("Usable endpoint(s):\n")
    for name, dims in working:
        print(f"  {name}  ({dims}-dim)")

    best, dims = working[0]
    print(
        f"\nTo switch to {best}:\n"
        f"  1. Set the ingest notebook's embedding_endpoint widget to {best!r}.\n"
        f"  2. Set DATABRICKS_EMBEDDING_ENDPOINT to {best!r} in app.yaml and redeploy.\n"
        f"  3. Re-run the ingest notebook.\n\n"
        f"Both steps matter: documents and queries must be embedded by the same\n"
        f"model or similarity scores are meaningless. The ingest notebook drops and\n"
        f"recreates the vector column automatically if the dimension changes\n"
        f"(this endpoint returns {dims})."
    )
