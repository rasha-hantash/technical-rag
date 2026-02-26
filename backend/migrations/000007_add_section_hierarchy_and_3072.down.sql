ALTER TABLE chunks DROP COLUMN IF EXISTS section_hierarchy;
ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE chunks ADD COLUMN embedding vector(1536);
