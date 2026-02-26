ALTER TABLE documents ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'processing';
ALTER TABLE documents ADD COLUMN file_size BIGINT;

-- Backfill existing documents as processed
UPDATE documents SET status = 'processed';

COMMENT ON COLUMN documents.status IS 'Ingestion lifecycle state: processing, processed, or error';
COMMENT ON COLUMN documents.file_size IS 'Size of the uploaded file in bytes';
