CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);

COMMENT ON TABLE documents IS 'Ingested PDF documents with deduplication via content hash';
COMMENT ON COLUMN documents.id IS 'Unique document identifier';
COMMENT ON COLUMN documents.file_hash IS 'SHA-256 hash of file contents for deduplication';
COMMENT ON COLUMN documents.file_path IS 'Original filename as uploaded by the user';
COMMENT ON COLUMN documents.metadata IS 'Arbitrary document metadata such as page count or author';
COMMENT ON COLUMN documents.created_at IS 'Timestamp when the document was first ingested';
