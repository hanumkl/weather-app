-- Raw NWS weather documents (alerts + forecast narratives)
CREATE TABLE IF NOT EXISTS weather_documents (
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
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);
