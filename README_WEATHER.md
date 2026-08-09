# Weather Intelligence — Unstructured Data → Lakebase Vector Search → REST API

Homework for **Day 2: Context Engineering with Databricks**.

Harvests unstructured weather text from the NWS API, vectorizes it into Lakebase (Postgres + pgvector), and exposes semantic search via a Flask REST API — all running on Databricks.

---

## Data Source

**National Weather Service API** — `https://api.weather.gov`

| Reason | Detail |
|---|---|
| Free / no API key | Focus on harvest → embed → retrieve, not auth plumbing |
| Rich unstructured text | Alert `description` + `instruction`, forecast `detailedForecast` |
| Stable public API | Well-documented GeoJSON; generous rate limits |

Geocoding uses a **static city → lat/lon map** (20 major US cities) so demos stay deterministic. Raw `lat,lon` pairs are also accepted.

---

## Schema

### `weather_documents` (raw NWS narratives)

| Column | Type | Purpose |
|---|---|---|
| `id` | TEXT PK | NWS alert id, or sha256 hash of location+grid+period for forecasts |
| `location` | TEXT | Display name (`Chicago, IL`) |
| `source_type` | TEXT | `alert` or `forecast` — filterable at search time |
| `headline` / `event` | TEXT | Short labels for UI / RAG context |
| `narrative_text` | TEXT | Free-text body that gets embedded |
| `issued_at` / `effective_at` | TIMESTAMPTZ | NWS timestamps |
| `payload` | JSONB | Raw JSON for provenance |
| `synced_at` | TIMESTAMPTZ | When we upserted the row |

### `weather_embeddings` (pgvector chunks)

| Column | Type | Purpose |
|---|---|---|
| `id` | TEXT PK | `{document_id}_{chunk_index}` |
| `document_id` | TEXT FK | References `weather_documents.id` |
| `chunk_index` | INT | Chunk position in document |
| `chunk_text` | TEXT | The chunked text |
| `embedding` | vector(1024) | 1024-dim embedding vector |
| `model_name` | TEXT | Model provenance |
| `created_at` | TIMESTAMPTZ | When embedded |

**Chunking:** `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (same as Day 2 news pipeline).

**Index:** `USING hnsw (embedding vector_cosine_ops)` for fast `<=>` queries.

**Upserts:** `ON CONFLICT (id) DO UPDATE` — re-running sync never duplicates rows.

### Embedding model: `databricks-gte-large-en` (1024-dim), not MiniLM

The assignment suggests `sentence-transformers/all-MiniLM-L6-v2` (384-dim) but allows a
different model if the dimensionality is documented. This project uses the Databricks
Foundation Model endpoint **`databricks-gte-large-en` → 1024 dimensions** for both
ingestion and query embedding.

Why:

1. **Both sides must share one vector space.** The Flask app embeds the incoming search
   query; the notebook embeds the documents. If those use different models, cosine
   similarity compares unrelated vector spaces and the ranking is meaningless. An earlier
   version of this project hit exactly that bug — MiniLM documents scored against a
   truncated BGE query vector produced results clustered in a meaningless 37–39% band.
2. **Databricks Apps can't host torch.** Apps are lightweight containers; `torch` is
   ~2.5GB and fails to install, so the app cannot run sentence-transformers locally. A
   shared serving endpoint is reachable from both the app and the cluster.
3. **No model download at runtime**, so notebook runs and app cold starts are faster.

The dimension is configurable via `EMBEDDING_DIM` + `DATABRICKS_EMBEDDING_ENDPOINT`
(set in `app.yaml` and the notebook widgets). The ingest notebook detects a mismatch
between the endpoint's output size and the existing `vector(N)` column, then drops and
recreates the table — stale vectors from a different model are not comparable.

---

## Project Layout

```
app.py                                  # Flask API (Databricks App)
app.yaml                                # Databricks App deployment config
lakebase.py                             # Lakebase connection (Databricks secrets)
weather_client.py                       # NWS API client + static geocode
embeddings.py                           # Chunking + SentenceTransformer helpers
setup_secrets.py                        # One-time: store Lakebase URL as secret
notebooks/
  ingest_weather_embeddings.py          # Databricks notebook: embed pipeline
  sync_weather.py                       # Databricks notebook: scheduled re-sync
  benchmark_hnsw.py                     # Databricks notebook: HNSW latency test
sql/
  01_weather_documents.sql              # DDL (also auto-created by lakebase.py)
  02_weather_embeddings.sql             # DDL with vector(384) + HNSW index
databricks.yml                          # Asset Bundle: scheduled sync+embed job
```

---

## Step-by-Step Setup (all on Databricks)

### 1. Create a Lakebase instance

1. In your Databricks workspace → **Catalog** → **Lakebase** tab.
2. Click **Create Lakebase instance**, give it a name, wait for **Available**.
3. Note the instance **name** — that name is the only thing you need to configure.

### 2. Point the app and notebook at that instance

Both connect as **your Databricks identity** using a credential minted per
connection, so no password is stored anywhere. Set the same instance name in
both places:

| Where | Setting |
|---|---|
| `app.yaml` | `LAKEBASE_INSTANCE_NAME` (then redeploy) |
| `ingest_weather_embeddings` | the `lakebase_instance` widget |

**These must match.** If they differ, the notebook writes embeddings to one
database while the app searches another, and `/weather/search` quietly returns
nothing. `GET /diagnostics` reports the instance, role and database the app is
actually using — check it there.

Not sure which instance to use? Run `notebooks/list_lakebase_instances`. It
lists every instance you can see and reports how many `weather_documents` and
`weather_embeddings` rows each already holds, so you can spot the one your
earlier sync wrote to.

Your Databricks identity needs a Postgres role on the instance; create one from
the instance's **Permissions** tab if `list_lakebase_instances` reports
`cannot inspect`.

<details>
<summary>Alternative: a native Postgres role with a password</summary>

Leave `LAKEBASE_INSTANCE_NAME` blank to fall back to a stored connection URL.
Enable **native password** auth under **Roles & Databases**, create a role, then
store the URL as the `database/lakebase-url` secret:

```python
%sh python setup_secrets.py
```

```
postgresql://role:password@host:5432/databricks_postgres?sslmode=require
```

Note that a role switch also needs `ALTER TABLE ... OWNER TO "<role>"`, run as
the previous owner, or the new role gets permission errors on existing tables.
</details>

### 3. Create a Git folder

1. **Workspace** → **Create** → **Git folder**.
2. Paste your GitHub repo URL (once you push this code).
3. Databricks clones the repo into your workspace.

### 4. Deploy the Flask app

1. **Compute** → **Apps** → **Create app** → **Custom**.
2. Point it at your Git folder (the one containing `app.py` and `app.yaml`).
3. Click **Deploy**. Databricks reads `app.yaml` automatically.
4. Once running, open the app URL and hit `GET /healthz` to confirm.

### 5. Sync weather data

From the deployed app URL (or via the Databricks App's built-in console):

```bash
curl -X POST https://<your-app-url>/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Austin, TX", "Seattle, WA"], "limit": 50}'
```

Expected response: `{"synced": 28, "locations": [...], "documents_fetched": 28}`

### 6. Run the embedding notebook

1. In your Git folder, open `notebooks/ingest_weather_embeddings.py`.
2. Attach it to a running cluster.
3. **Run All** — it will:
   - Uninstall `psycopg2` / `psycopg2-binary` (see note below)
   - Read unembedded docs from `weather_documents`
   - Chunk and embed them via `databricks-gte-large-en` (1024-dim)
   - Write vectors into `weather_embeddings` via `execute_values` + `::vector`
   - Verify with a sample similarity query

Widgets let you override the endpoint, chunk size/overlap, and set
`rebuild_all=true` to re-embed every document instead of only new ones.

> **Why uninstall psycopg2?** The Databricks runtime already ships `psycopg2`.
> A pip-installed copy alongside it crashes the kernel with
> *"Fatal error: The Python kernel is unresponsive."* All three notebooks
> uninstall it first and rely on the runtime's version.

### 7. Semantic search

```bash
curl -X POST https://<your-app-url>/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "flash flood risk this weekend", "top_k": 5}'
```

Filter by source type:
```bash
curl -X POST https://<your-app-url>/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "severe thunderstorm", "top_k": 5, "source_type": "alert"}'
```

RAG summary (stretch):
```bash
curl 'https://<your-app-url>/weather/search?query=flooding%20near%20rivers&top_k=5&summarize=true'
```

---

## Stretch Goals Included

| Goal | Implementation |
|---|---|
| Upsert / dedupe | `ON CONFLICT (id) DO UPDATE` in sync |
| Alerts + forecasts + `source_type` filter | Both harvested; search accepts `source_type` param |
| GET search + LLM summary (RAG) | `GET /weather/search?summarize=true` — uses Databricks Foundation Model, extractive fallback |
| Scheduled re-sync | `databricks.yml` defines a 2-task job (sync → embed, every 30m, paused by default) |
| HNSW benchmark | `notebooks/benchmark_hnsw.py` — drops/recreates index, compares latency |

### Schedule the job (Databricks Asset Bundle)

```bash
# Set workspace.host in databricks.yml first
databricks bundle deploy -t dev
databricks bundle run weather_sync_embed_job -t dev
# After a successful run, flip pause_status to UNPAUSED and redeploy
```

Or create the job manually via **Workflows UI** → **Create Job** with two notebook tasks.

---

## API Reference

| Method | Path | Body / Query |
|---|---|---|
| GET | `/healthz` | — |
| POST | `/weather/sync` | `{"locations": ["Chicago, IL"], "limit": 50}` |
| POST | `/weather/search` | `{"query": "...", "top_k": 5, "source_type": "alert"}` |
| GET | `/weather/search` | `?query=...&top_k=5&source_type=forecast&summarize=true` |
| GET | `/weather/documents` | `?limit=50&source_type=alert` |

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| App deploy fails | Don't put `torch` / `sentence-transformers` in `requirements.txt` — too large for Apps |
| Notebook: "Python kernel is unresponsive" | A pip-installed psycopg2 conflicts with the runtime's. Notebooks uninstall it first |
| Search returns an error about dimensions | Endpoint output size ≠ `vector(N)` column. Align `EMBEDDING_DIM` and re-run the notebook |
| RAG summary looks templated | The LLM call failed and fell back to extractive. Check `GET /diagnostics` → `llm_query_test` |
| All similarity scores in a narrow band | Query and documents were embedded by different models — re-run the notebook |
| Notebook: `TimeoutError: Timed out after 0:05:00` | `serving_endpoints.query()` retries internally for 5 min with no per-request timeout. Both notebook and app now call the REST `invocations` API with an explicit timeout |
| `UndefinedTable: relation "weather_documents" does not exist` | The tables live in a different Lakebase instance, or none has been created yet. Confirm the app and notebook use the same `LAKEBASE_INSTANCE_NAME` / `lakebase_instance`, then run `POST /weather/sync` |
| Search returns nothing the notebook clearly wrote | App and notebook are pointed at different Lakebase instances. Compare `GET /diagnostics` against the notebook's `lakebase_instance` widget |
| `permission denied for table weather_documents` | The connecting Postgres role is not the table owner. Changing the role in `database/lakebase-url` requires `ALTER TABLE ... OWNER TO "<role>"` first, run as the **old** owner. Ownership (not just `GRANT ALL`) is needed because the ingest notebook may drop and recreate the embeddings table during a dimension migration |
| `429 REQUEST_LIMIT_EXCEEDED` | Shared workspace request budget for pay-per-token endpoints is saturated. Runs are resumable, so re-run to continue; or switch `embedding_endpoint`. See below |
| Fix pushed to git but notebook behaves the same | Databricks runs its own copy. Pull in the Git folder, then `dbutils.widgets.removeAll()` + re-run cell 1, since **widget values persist and ignore new code defaults**. Check the `Code version:` line to confirm |

`GET /diagnostics` reports the configured endpoints, which ones this app can
actually see, the live LLM test result, and the table's declared vector width.

### Rate limits (important)

Pay-per-token Foundation Model endpoints share one **workspace-wide** request
budget, so a busy workspace (e.g. a whole bootcamp cohort on
`databricks-gte-large-en`) can exhaust it. Exceeding it returns
`429 REQUEST_LIMIT_EXCEEDED`. Because the limit counts *requests* rather than
tokens, throughput comes from **fewer, larger, spaced-out requests**;
the embedding path is deliberately sequential:

| Widget | Default | Why |
|---|---|---|
| `request_batch` | 32 | Many chunks per request → few requests total |
| `sleep_between` | 1.0 | Stays inside the per-second window |
| `request_timeout` | 60 | Explicit, so a stuck call fails fast |
| `max_attempts` | 8 | Rides out a temporarily saturated endpoint |

On 429 the notebook backs off 10s, 20s, 40s (capped at 60s) with jitter and
honours `Retry-After`. Short 1–2s retries do not clear the window. Oversized
payloads are split in half automatically.

**Progress is resumable.** Each batch is committed to Lakebase as soon as it is
embedded, and the document query only selects chunks with no embedding yet, so a
429 partway through keeps everything already written — just re-run the notebook.

If the budget is exhausted by activity outside your control, no client-side
tuning helps. Run `notebooks/probe_embedding_endpoints` to see which endpoints
respond right now — it sends one short request to each and reports status,
latency and dimensions. Then either wait for capacity, or point
`embedding_endpoint` at a less contended endpoint. `databricks-bge-large-en` is also 1024-dim, so the schema
still fits — but set `DATABRICKS_EMBEDDING_ENDPOINT` in `app.yaml` to match and
re-embed, since queries and documents must share one vector space.

The app retries only briefly (3 attempts) since a user is waiting on the response.

## Known Limitations

- **Databricks Free Edition throttles the embedding endpoint.** Foundation Model
  API limits are published for Enterprise tier only and "vary based on the
  workspace platform tier"; Free Edition sits well below them. For embedding
  models the binding limit is **queries per hour**, not tokens or QPS, so the
  fix is fewer requests rather than slower ones — `request_batch=128` sends this
  corpus as a single call. When the hourly budget is already spent, no
  client-side retry can clear it inside a notebook run; re-run once the window
  rolls. `databricks-bge-large-en` carries a 4x larger QPH allowance and is also
  1024-dim, so it substitutes into this schema without a migration.
  Given more time, the durable fix on Free Edition is to drop the pay-per-token
  dependency entirely: embed locally with `all-MiniLM-L6-v2` (384-dim) in the
  notebook and serve query embeddings from an ONNX runtime in the app, which
  removes both the quota and the torch-in-Apps problem.
- Static geocode covers ~20 US cities; others need `lat,lon` format.
- NWS alerts are sparse in calm weather — forecasts keep the corpus populated.
- Embedding calls go one batch at a time; fine for homework volumes, would want
  concurrency for large corpora.
- RAG summary needs a queryable chat endpoint; falls back to extractive without one.
- HNSW speedup is modest on small tables.
- Deviates from the assignment's suggested MiniLM model (see the embedding model
  section above for the reasoning).
