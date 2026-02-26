ALTER TABLE chunks ADD COLUMN section_hierarchy TEXT;

-- Migrate embedding dimension from 1536 to 3072 for text-embedding-3-large.
-- Requires drop+recreate since ALTER COLUMN cannot change vector dimensions.
-- Safe for a fresh fork with no existing data.
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE chunks ADD COLUMN embedding vector(3072);

-- pgvector HNSW/IVFFlat indexes max out at 2000 dimensions (8KB page limit).
-- For 3072-dimension embeddings, skip the index — exact brute-force search is
-- fast enough for <50k chunks (~20 books). For larger datasets, add a halfvec-based
-- index later: CREATE INDEX ... USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);
DROP INDEX IF EXISTS chunks_embedding_idx;
