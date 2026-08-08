"""
National Weather Service (api.weather.gov) client.

No API key required. NWS asks for a descriptive User-Agent so they can
contact you if something goes wrong — set NWS_USER_AGENT in .env.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger("weather-app.weather_client")

_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov").rstrip("/")
_DEFAULT_TIMEOUT = 30
_DEFAULT_UA = os.environ.get(
    "NWS_USER_AGENT",
    "weather-app-bootcamp (student@example.com)",
)

# Static city → (lat, lon, display_name) map for homework demos.
# Keys are normalized as "city, st" (lowercase, single space after comma).
CITY_COORDS: dict[str, tuple[float, float, str]] = {
    "chicago, il": (41.8781, -87.6298, "Chicago, IL"),
    "austin, tx": (30.2672, -97.7431, "Austin, TX"),
    "new york, ny": (40.7128, -74.0060, "New York, NY"),
    "los angeles, ca": (34.0522, -118.2437, "Los Angeles, CA"),
    "seattle, wa": (47.6062, -122.3321, "Seattle, WA"),
    "miami, fl": (25.7617, -80.1918, "Miami, FL"),
    "denver, co": (39.7392, -104.9903, "Denver, CO"),
    "boston, ma": (42.3601, -71.0589, "Boston, MA"),
    "atlanta, ga": (33.7490, -84.3880, "Atlanta, GA"),
    "phoenix, az": (33.4484, -112.0740, "Phoenix, AZ"),
    "dallas, tx": (32.7767, -96.7970, "Dallas, TX"),
    "houston, tx": (29.7604, -95.3698, "Houston, TX"),
    "san francisco, ca": (37.7749, -122.4194, "San Francisco, CA"),
    "portland, or": (45.5152, -122.6784, "Portland, OR"),
    "minneapolis, mn": (44.9778, -93.2650, "Minneapolis, MN"),
    "detroit, mi": (42.3314, -83.0458, "Detroit, MI"),
    "philadelphia, pa": (39.9526, -75.1652, "Philadelphia, PA"),
    "washington, dc": (38.9072, -77.0369, "Washington, DC"),
    "nashville, tn": (36.1627, -86.7816, "Nashville, TN"),
    "new orleans, la": (29.9511, -90.0715, "New Orleans, LA"),
}


def _normalize_location(location: str) -> str:
    return re.sub(r"\s+", " ", location.strip().lower())


def resolve_location(location: str) -> tuple[float, float, str]:
    """
    Resolve a "City, ST" string to (lat, lon, display_name).

    Supports:
    - Static map lookups (preferred for homework)
    - Raw "lat,lon" pairs (e.g. "41.8781,-87.6298")
    """
    key = _normalize_location(location)
    if key in CITY_COORDS:
        return CITY_COORDS[key]

    # Accept "lat,lon" directly
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", location)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        return lat, lon, f"{lat},{lon}"

    known = ", ".join(sorted({v[2] for v in CITY_COORDS.values()}))
    raise ValueError(
        f"Unknown location {location!r}. Use a known city "
        f"(e.g. {known}) or a lat,lon pair."
    )


def _stable_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _parse_ts(value: str | None) -> str | None:
    if not value:
        return None
    try:
        # NWS returns ISO-8601 with offset; fromisoformat handles most of it
        # once we normalize a trailing Z.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return value


class WeatherClient:
    """Thin NWS API wrapper: points → alerts + forecast narratives."""

    def __init__(
        self,
        base_url: str | None = None,
        user_agent: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent or _DEFAULT_UA,
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_point(self, lat: float, lon: float) -> dict[str, Any]:
        """GET /points/{lat},{lon} → grid office / x / y + forecast URLs."""
        # NWS wants at most 4 decimal places
        return self.get(f"/points/{lat:.4f},{lon:.4f}")

    def get_active_alerts_for_point(self, lat: float, lon: float) -> list[dict]:
        data = self.get("/alerts/active", params={"point": f"{lat:.4f},{lon:.4f}"})
        return data.get("features", []) or []

    def get_forecast(self, forecast_url: str) -> dict[str, Any]:
        return self.get(forecast_url)

    def harvest_location(self, location: str, limit: int = 50) -> list[dict]:
        """
        Resolve a location, fetch alerts + forecast periods, normalize to
        document records ready for upsert into weather_documents.
        """
        lat, lon, display = resolve_location(location)
        point = self.get_point(lat, lon)
        props = point.get("properties") or {}
        forecast_url = props.get("forecast")
        office = props.get("gridId")
        grid_x = props.get("gridX")
        grid_y = props.get("gridY")

        docs: list[dict] = []
        synced_at = datetime.now(timezone.utc).isoformat()

        # --- Alerts ---
        try:
            alerts = self.get_active_alerts_for_point(lat, lon)
        except requests.HTTPError as exc:
            logger.warning("Alerts fetch failed for %s: %s", display, exc)
            alerts = []

        for feature in alerts:
            doc = self._normalize_alert(feature, display, synced_at)
            if doc:
                docs.append(doc)
            if len(docs) >= limit:
                return docs[:limit]

        # --- Forecast periods ---
        if forecast_url:
            try:
                forecast = self.get_forecast(forecast_url)
                periods = (forecast.get("properties") or {}).get("periods") or []
                for period in periods:
                    doc = self._normalize_forecast(
                        period,
                        display=display,
                        office=office,
                        grid_x=grid_x,
                        grid_y=grid_y,
                        synced_at=synced_at,
                        raw_parent=forecast,
                    )
                    if doc:
                        docs.append(doc)
                    if len(docs) >= limit:
                        break
            except requests.HTTPError as exc:
                logger.warning("Forecast fetch failed for %s: %s", display, exc)

        return docs[:limit]

    def harvest_locations(
        self, locations: list[str], limit_per_location: int = 50
    ) -> list[dict]:
        """Harvest documents for many locations (deduped by document id)."""
        seen: set[str] = set()
        out: list[dict] = []
        for loc in locations:
            for doc in self.harvest_location(loc, limit=limit_per_location):
                if doc["id"] in seen:
                    continue
                seen.add(doc["id"])
                out.append(doc)
        return out

    def _normalize_alert(
        self, feature: dict, display: str, synced_at: str
    ) -> dict | None:
        props = feature.get("properties") or {}
        alert_id = props.get("id") or feature.get("id")
        if not alert_id:
            return None

        headline = props.get("headline") or props.get("event") or "Weather Alert"
        event = props.get("event")
        description = (props.get("description") or "").strip()
        instruction = (props.get("instruction") or "").strip()
        narrative_parts = [p for p in (description, instruction) if p]
        narrative = "\n\n".join(narrative_parts).strip()
        if not narrative:
            narrative = headline

        return {
            "id": str(alert_id),
            "location": display,
            "source_type": "alert",
            "headline": headline,
            "event": event,
            "narrative_text": narrative,
            "issued_at": _parse_ts(props.get("sent") or props.get("onset")),
            "effective_at": _parse_ts(props.get("effective") or props.get("onset")),
            "payload": feature,
            "synced_at": synced_at,
        }

    def _normalize_forecast(
        self,
        period: dict,
        *,
        display: str,
        office: str | None,
        grid_x: Any,
        grid_y: Any,
        synced_at: str,
        raw_parent: dict,
    ) -> dict | None:
        name = period.get("name") or "Forecast"
        start = period.get("startTime") or ""
        detailed = (period.get("detailedForecast") or period.get("shortForecast") or "").strip()
        if not detailed:
            return None

        # Stable id: location + grid + period start (or name fallback)
        id_key = _stable_id(
            display,
            str(office or ""),
            str(grid_x or ""),
            str(grid_y or ""),
            start or name,
            "forecast",
        )
        headline = f"{display}: {name}"
        event = period.get("shortForecast")

        return {
            "id": id_key,
            "location": display,
            "source_type": "forecast",
            "headline": headline,
            "event": event,
            "narrative_text": detailed,
            "issued_at": _parse_ts(start) or synced_at,
            "effective_at": _parse_ts(start),
            "payload": {"period": period, "grid": {"office": office, "x": grid_x, "y": grid_y}},
            "synced_at": synced_at,
        }
