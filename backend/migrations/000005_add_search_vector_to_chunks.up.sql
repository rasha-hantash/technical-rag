-- Add a generated tsvector column for full-text search (BM25-style)
ALTER TABLE chunks ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- GIN index for fast full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON chunks USING GIN (search_vector);

COMMENT ON COLUMN chunks.search_vector IS 'Auto-generated tsvector from content column for BM25/full-text search. Uses English dictionary.';
