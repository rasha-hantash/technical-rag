DROP INDEX IF EXISTS idx_documents_title;
DROP INDEX IF EXISTS idx_documents_author;
DROP INDEX IF EXISTS idx_documents_publication_year;

ALTER TABLE documents DROP COLUMN IF EXISTS title;
ALTER TABLE documents DROP COLUMN IF EXISTS author;
ALTER TABLE documents DROP COLUMN IF EXISTS edition;
ALTER TABLE documents DROP COLUMN IF EXISTS publication_year;
