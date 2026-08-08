# Databricks notebook source
# MAGIC %md
# MAGIC # Sync Weather Documents from NWS
# MAGIC
# MAGIC Stretch goal: scheduled re-sync. Run this notebook as a Databricks Job
# MAGIC (every 30 minutes) to keep `weather_documents` fresh. Pair it with
# MAGIC `ingest_weather_embeddings` to auto-embed new documents.
# MAGIC
# MAGIC This does the same thing as `POST /weather/sync` but from a notebook
# MAGIC so it can be scheduled as a Workflow task.

# COMMAND ----------

# MAGIC %pip install psycopg2-binary requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("locations", "Chicago, IL;Austin, TX;Seattle, WA", "Locations (semicolon-separated)")
dbutils.widgets.text("limit", "50", "Max docs per location")

LOCATIONS = [loc.strip() for loc in dbutils.widgets.get("locations").split(";") if loc.strip()]
LIMIT = int(dbutils.widgets.get("limit"))

print(f"Locations: {LOCATIONS}")
print(f"Limit per location: {LIMIT}")

# COMMAND ----------

# DBTITLE 1,Connect to Lakebase
import base64
import json
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

DOCUMENTS_TABLE = "weather_documents"

# Ensure table exists
conn = get_conn()
cur = conn.cursor()
cur.execute(f"""
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
conn.commit()
cur.close()
conn.close()
print("Table ready")

# COMMAND ----------

# DBTITLE 1,Harvest from NWS
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

from weather_client import WeatherClient

client = WeatherClient()
docs = client.harvest_locations(LOCATIONS, limit_per_location=LIMIT)
print(f"Fetched {len(docs)} documents from NWS")
for d in docs[:5]:
    print(f"  [{d['source_type']}] {d['location']}: {d.get('headline', '')[:60]}")

# COMMAND ----------

# DBTITLE 1,Upsert into Lakebase
count = 0
conn = get_conn()
cur = conn.cursor()
for doc in docs:
    cur.execute(f"""
        INSERT INTO {DOCUMENTS_TABLE} (
            id, location, source_type, headline, event,
            narrative_text, issued_at, effective_at, payload, synced_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
        ON CONFLICT (id) DO UPDATE SET
            location = EXCLUDED.location,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            event = EXCLUDED.event,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            effective_at = EXCLUDED.effective_at,
            payload = EXCLUDED.payload,
            synced_at = now()
    """, (
        doc["id"], doc["location"], doc["source_type"],
        doc.get("headline"), doc.get("event"), doc["narrative_text"],
        doc.get("issued_at"), doc.get("effective_at"),
        json.dumps(doc.get("payload") or {}),
    ))
    count += 1
conn.commit()
cur.close()
conn.close()

print(f"Upserted {count} documents into {DOCUMENTS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC Done! Run `ingest_weather_embeddings` next to embed any new documents.
