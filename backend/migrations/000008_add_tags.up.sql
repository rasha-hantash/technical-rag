-- Add tags column to documents for topic-based categorization and scoped queries.
-- Tags are required at upload time (enforced by application, not DB constraint,
-- since existing rows need the empty-array default for migration).
ALTER TABLE documents ADD COLUMN tags TEXT[] NOT NULL DEFAULT '{}';

-- GIN index for fast array-overlap queries: WHERE d.tags && ARRAY['rust']
CREATE INDEX idx_documents_tags ON documents USING gin (tags);
