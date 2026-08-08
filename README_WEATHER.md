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
| `embedding` | vector(384) | 384-dim embedding vector |
| `model_name` | TEXT | Model provenance |
| `created_at` | TIMESTAMPTZ | When embedded |

**Chunking:** `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100` (same as Day 2 news pipeline).

**Model:** `sentence-transformers/all-MiniLM-L6-v2` → **384 dimensions**.

**Index:** `USING hnsw (embedding vector_cosine_ops)` for fast `<=>` queries.

**Upserts:** `ON CONFLICT (id) DO UPDATE` — re-running sync never duplicates rows.

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
3. Go to **Roles & Databases** → enable **native password** auth.
4. **Create a new role** with password auth. Copy the connection URL:
   ```
   postgresql://role:password@host:5432/databricks_postgres?sslmode=require
   ```

### 2. Store the secret

1. Open or create a notebook in your Databricks workspace.
2. Run in a cell:
   ```python
   %sh python setup_secrets.py
   ```
   Paste the Lakebase URL when prompted. This stores it as `database/lakebase-url`.

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
   - Install `sentence-transformers` and `psycopg2-binary`
   - Read unembedded docs from `weather_documents`
   - Chunk and embed them (384-dim)
   - Write vectors into `weather_embeddings` via `execute_values` + `::vector`
   - Verify with a sample similarity query

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

## Known Limitations

- Static geocode covers ~20 US cities; others need `lat,lon` format.
- NWS alerts are sparse in calm weather — forecasts keep the corpus populated.
- First search downloads MiniLM weights (~90MB); cold start takes ~1 min.
- RAG summary needs a Databricks model serving endpoint; falls back to extractive without one.
- HNSW speedup is modest on small tables.
