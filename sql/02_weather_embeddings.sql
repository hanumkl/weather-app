-- Chunk embeddings for weather_documents (384-dim = all-MiniLM-L6-v2)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);

-- HNSW index for cosine similarity (pgvector <=> operator)
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);
