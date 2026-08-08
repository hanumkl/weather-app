"""
Weather Intelligence Flask API — Databricks App.

Pipeline:
  POST /weather/sync   → harvest NWS text → weather_documents
  (run notebooks/ingest_weather_embeddings.py on cluster) → weather_embeddings
  POST /weather/search → semantic search over pgvector
  GET  /weather/search → same + optional LLM summary (RAG stretch)

Deploy as a Databricks App using app.yaml.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from flask import Flask, jsonify, render_template, request

import lakebase
from embeddings import embed_query, vector_literal, warm_model
from weather_client import CITY_COORDS, WeatherClient, resolve_location

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)

DEFAULT_LOCATIONS = [
    loc.strip()
    for loc in os.environ.get(
        "DEFAULT_WEATHER_LOCATIONS", "Chicago, IL;Austin, TX;Seattle, WA"
    ).split(";")
    if loc.strip()
]

DOCUMENTS_TABLE = lakebase.DOCUMENTS_TABLE
EMBEDDINGS_TABLE = lakebase.EMBEDDINGS_TABLE


@app.before_request
def _ensure_schema_once():
    """Lazily create tables on the first request (cheap IF NOT EXISTS)."""
    if getattr(app, "_schema_ready", False):
        return
    try:
        lakebase.ensure_schema()
        app._schema_ready = True
    except Exception:  # noqa: BLE001
        logger.exception("Schema ensure failed (will retry next request)")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    logger.exception("Unhandled exception")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Browser UI for running the sync → embed → search pipeline."""
    cities = sorted({display for _, _, display in CITY_COORDS.values()})
    return render_template("index.html", cities=cities)


@app.route("/api")
def api_index():
    return jsonify(
        {
            "service": "weather-intelligence",
            "endpoints": {
                "GET /healthz": "health check",
                "GET /diagnostics": "config + serving endpoints this app can see",
                "POST /weather/sync": "harvest NWS alerts+forecasts into Lakebase",
                "POST /weather/search": "semantic search over weather_embeddings",
                "GET /weather/search": "search + optional LLM summary (RAG)",
                "GET /weather/documents": "list raw synced documents",
            },
        }
    )


@app.route("/diagnostics")
def diagnostics():
    """
    Report which model serving endpoints this app's service principal can see.

    Useful for picking a value for DATABRICKS_LLM_ENDPOINT: the app runs as its
    own service principal, so it may see a different set than you do in the UI.
    """
    configured_llm = os.environ.get(
        "DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
    )
    info: dict[str, Any] = {
        "configured_llm_endpoint": configured_llm,
        "configured_embedding_endpoint": os.environ.get(
            "DATABRICKS_EMBEDDING_ENDPOINT", "databricks-bge-large-en"
        ),
    }

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        names = sorted(ep.name for ep in w.serving_endpoints.list() if ep.name)
        info["visible_serving_endpoints"] = names
        info["configured_llm_is_visible"] = configured_llm in names
        info["chat_like_endpoints"] = [
            n
            for n in names
            if any(k in n.lower() for k in ("llama", "gpt", "claude", "mixtral", "dbrx", "qwen", "gemma"))
        ]
        info["embedding_like_endpoints"] = [
            n for n in names if any(k in n.lower() for k in ("bge", "embed", "gte"))
        ]
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"Could not list serving endpoints: {exc}"

    # Embedding config (shared by the notebook and this app)
    from embeddings import describe_backend

    info["embedding"] = describe_backend()

    # Confirm the vector column width matches what the endpoint returns
    try:
        dim_rows = lakebase.run_query(
            """
            SELECT a.atttypmod AS declared_dim
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            WHERE c.relname = %s AND a.attname = 'embedding'
            """,
            (EMBEDDINGS_TABLE,),
        )
        if dim_rows:
            info["embedding"]["table_vector_dim"] = dim_rows[0]["declared_dim"]
    except Exception as exc:  # noqa: BLE001
        info["embedding"]["table_vector_dim_error"] = str(exc)

    # Actually try the chat endpoint so failures aren't silent
    try:
        reply = query_chat_endpoint(configured_llm, "Reply with the single word: ok")
        info["llm_query_test"] = {"ok": True, "reply": reply[:100]}
    except Exception as exc:  # noqa: BLE001
        info["llm_query_test"] = {"ok": False, "error": str(exc)}

    return jsonify(info)


@app.route("/weather/documents")
def list_documents():
    """List recently synced weather documents (debug / sanity check)."""
    lakebase.ensure_weather_documents_table()
    limit = max(1, min(int(request.args.get("limit", 50)), 200))
    source_type = request.args.get("source_type")
    sql = (
        f"SELECT id, location, source_type, headline, event, "
        f"LEFT(narrative_text, 200) AS narrative_preview, "
        f"issued_at, synced_at "
        f"FROM {DOCUMENTS_TABLE}"
    )
    params: list[Any] = []
    if source_type in ("alert", "forecast"):
        sql += " WHERE source_type = %s"
        params.append(source_type)
    sql += " ORDER BY synced_at DESC LIMIT %s"
    params.append(limit)
    return jsonify(lakebase.run_query(sql, tuple(params)))


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Harvest NWS alerts + forecasts for locations and upsert into weather_documents.

    Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}
    """
    lakebase.ensure_weather_documents_table()

    body = request.get_json(silent=True) or {}
    locations = body.get("locations") or DEFAULT_LOCATIONS
    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "locations must be a non-empty list of strings"}), 400

    cleaned: list[str] = []
    for loc in locations:
        if not isinstance(loc, str) or not loc.strip():
            continue
        try:
            resolve_location(loc)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        cleaned.append(loc.strip())

    if not cleaned:
        return jsonify({"error": "no valid locations provided"}), 400

    limit = int(body.get("limit", 50))
    limit = max(1, min(limit, 200))

    client = WeatherClient()
    docs = client.harvest_locations(cleaned, limit_per_location=limit)
    upserted = _upsert_weather_documents(docs)

    return jsonify(
        {
            "synced": upserted,
            "locations": cleaned,
            "documents_fetched": len(docs),
        }
    )


@app.route("/weather/search", methods=["POST"])
def search_weather_post():
    """
    Semantic search over weather_embeddings.

    Body: {
      "query": "flash flood risk this weekend",
      "top_k": 5,
      "source_type": "alert" | "forecast" | null
    }
    """
    body = request.get_json(silent=True) or {}
    query = body.get("query")
    top_k = body.get("top_k", 5)
    source_type = body.get("source_type")
    return _search_response(query=query, top_k=top_k, source_type=source_type, summarize=False)


@app.route("/weather/search", methods=["GET"])
def search_weather_get():
    """
    GET variant with RAG summary:
      /weather/search?query=...&top_k=5&source_type=alert&summarize=true
    """
    query = request.args.get("query")
    top_k = request.args.get("top_k", 5)
    source_type = request.args.get("source_type")
    summarize = str(request.args.get("summarize", "true")).lower() in (
        "1",
        "true",
        "yes",
    )
    return _search_response(
        query=query, top_k=top_k, source_type=source_type, summarize=summarize
    )


def _search_response(
    *,
    query: Any,
    top_k: Any,
    source_type: Any,
    summarize: bool,
):
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "query is required and must be a non-empty string"}), 400

    try:
        top_k_int = int(top_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400
    top_k_int = max(1, min(top_k_int, 20))

    if source_type is not None and source_type not in ("alert", "forecast", ""):
        return jsonify({"error": "source_type must be 'alert', 'forecast', or omitted"}), 400
    if source_type == "":
        source_type = None

    lakebase.ensure_schema()

    count_rows = lakebase.run_query(f"SELECT COUNT(*) AS n FROM {EMBEDDINGS_TABLE}")
    if not count_rows or int(count_rows[0]["n"]) == 0:
        return jsonify(
            {
                "query": query.strip(),
                "top_k": top_k_int,
                "results": [],
                "message": (
                    "weather_embeddings is empty. Run POST /weather/sync then "
                    "run notebooks/ingest_weather_embeddings.py on a cluster first."
                ),
            }
        )

    try:
        warm_model()
        qvec = embed_query(query.strip())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to embed query")
        return jsonify({"error": f"embedding failed: {exc}"}), 500

    vec = vector_literal(qvec)
    filter_sql = ""
    if source_type in ("alert", "forecast"):
        filter_sql = "WHERE d.source_type = %s"
        params: list[Any] = [vec, source_type, vec, top_k_int]
    else:
        params = [vec, vec, top_k_int]

    sql = f"""
        SELECT
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.event,
            d.narrative_text,
            e.chunk_text,
            e.chunk_index,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        {filter_sql}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """
    rows = lakebase.run_query(sql, tuple(params))
    results = [
        {
            "id": r["id"],
            "location": r["location"],
            "source_type": r["source_type"],
            "headline": r["headline"],
            "event": r.get("event"),
            "chunk_text": r["chunk_text"],
            "chunk_index": r["chunk_index"],
            "similarity": float(r["similarity"]) if r["similarity"] is not None else None,
        }
        for r in rows
    ]

    payload: dict[str, Any] = {
        "query": query.strip(),
        "top_k": top_k_int,
        "source_type": source_type,
        "results": results,
    }
    if summarize:
        payload["summary"] = _summarize_results(query.strip(), results)
    return jsonify(payload)


def _upsert_weather_documents(docs: list[dict]) -> int:
    """Upsert documents by id so re-syncs do not create duplicates."""
    if not docs:
        return 0

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in docs:
                cur.execute(
                    f"""
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
                    """,
                    (
                        doc["id"],
                        doc["location"],
                        doc["source_type"],
                        doc.get("headline"),
                        doc.get("event"),
                        doc["narrative_text"],
                        doc.get("issued_at"),
                        doc.get("effective_at"),
                        json.dumps(doc.get("payload") or {}),
                    ),
                )
                count += 1
        conn.commit()
    return count


def query_chat_endpoint(endpoint: str, prompt: str) -> str:
    """
    Call a Databricks chat serving endpoint and return the text response.

    The SDK expects ChatMessage objects here, not plain dicts — passing dicts
    fails during serialization.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    w = WorkspaceClient()
    response = w.serving_endpoints.query(
        name=endpoint,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="Answer briefly using only the provided context.",
            ),
            ChatMessage(role=ChatMessageRole.USER, content=prompt),
        ],
        max_tokens=300,
        temperature=0.2,
    )

    if response.choices:
        message = response.choices[0].message
        if message is not None and message.content:
            return str(message.content).strip()

    raise RuntimeError(f"Endpoint {endpoint!r} returned no message content")


def _summarize_results(query: str, results: list[dict]) -> str:
    """
    RAG summary: Databricks Foundation Model when available, extractive fallback otherwise.
    """
    if not results:
        return "No matching weather documents were found for that query."

    context_blocks = []
    for i, r in enumerate(results[:5], start=1):
        context_blocks.append(
            f"[{i}] ({r.get('source_type')}) {r.get('location')} — "
            f"{r.get('headline')}\n{r.get('chunk_text')}"
        )
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are a helpful weather briefing assistant. Using ONLY the retrieved "
        "weather documents below, write a short (3-5 sentence) natural-language "
        f"summary answering the user question.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved documents:\n{context}\n\n"
        "Summary:"
    )

    endpoint = os.environ.get(
        "DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
    )
    try:
        return query_chat_endpoint(endpoint, prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM summary unavailable (%s); using extractive fallback", exc)

    lines = [f"Based on {len(results)} retrieved weather documents for \u201c{query}\u201d:"]
    for r in results[:3]:
        snippet = (r.get("chunk_text") or "").split(".")[0].strip()
        lines.append(f"- {r.get('location')}: {r.get('headline')} \u2014 {snippet}.")
    return " ".join(lines)


if __name__ == "__main__":
    try:
        warm_model()
    except Exception:  # noqa: BLE001
        logger.warning(
            "Could not pre-load embedding model at startup; "
            "it will load on first /weather/search call."
        )

    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "8000"))
    app.run(debug=True, host=host, port=port)
