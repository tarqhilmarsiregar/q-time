-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Academic documents
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,

    title VARCHAR(255) NOT NULL,
    file_name VARCHAR(255) NOT NULL,

    category VARCHAR(100) NOT NULL,
    academic_year VARCHAR(20),

    version VARCHAR(50),
    effective_date DATE,

    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'published', 'archived')),

    content_sha256 CHAR(64) UNIQUE NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document chunks and embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,

    document_id BIGINT NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,

    content TEXT NOT NULL,

    embedding VECTOR(384) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(document_id, chunk_index)
);

-- Query optimization for metadata filters
CREATE INDEX IF NOT EXISTS idx_documents_status_category
ON documents(status, category);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON document_chunks(document_id);
