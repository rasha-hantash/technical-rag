CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_type VARCHAR(50),
    page_number INTEGER,
    position INTEGER,
    embedding vector(1536),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
-- NOTE: IVFFlat index should be created after data ingestion, not on empty tables.
-- See: TODO - Add migration for vector index after initial data load (requires lists * 1000 rows minimum).

COMMENT ON TABLE chunks IS 'Text chunks extracted from documents with optional vector embeddings for similarity search';
COMMENT ON COLUMN chunks.id IS 'Unique chunk identifier';
COMMENT ON COLUMN chunks.document_id IS 'Parent document this chunk was extracted from';
COMMENT ON COLUMN chunks.content IS 'Raw text content of the chunk';
COMMENT ON COLUMN chunks.chunk_type IS 'Classification of chunk content, e.g. text, table, heading';
COMMENT ON COLUMN chunks.page_number IS '1-based page number where this chunk appears in the PDF';
COMMENT ON COLUMN chunks.position IS 'Sequential ordering of this chunk within the document';
COMMENT ON COLUMN chunks.embedding IS '1536-dimensional vector embedding for semantic similarity search';
COMMENT ON COLUMN chunks.created_at IS 'Timestamp when the chunk was created';
