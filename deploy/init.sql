CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS proptech_docs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL,
    title TEXT,
    url TEXT,
    chunk_text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS proptech_docs_embedding_idx 
ON proptech_docs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS proptech_docs_source_idx ON proptech_docs(source);
