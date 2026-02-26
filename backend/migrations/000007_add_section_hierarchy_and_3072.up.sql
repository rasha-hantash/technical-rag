ALTER TABLE chunks ADD COLUMN section_hierarchy TEXT;

-- Migrate embedding dimension from 1536 to 3072 for text-embedding-3-large.
-- Requires drop+recreate since ALTER COLUMN cannot change vector dimensions.
-- Safe for a fresh fork with no existing data.
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE chunks ADD COLUMN embedding vector(3072);
