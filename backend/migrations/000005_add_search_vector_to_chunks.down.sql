DROP INDEX IF EXISTS idx_chunks_search_vector;
ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector;
