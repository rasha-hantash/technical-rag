DROP INDEX IF EXISTS idx_documents_tags;
ALTER TABLE documents DROP COLUMN IF EXISTS tags;
