ALTER TABLE documents ADD COLUMN title TEXT;
ALTER TABLE documents ADD COLUMN author TEXT;
ALTER TABLE documents ADD COLUMN edition VARCHAR(50);
ALTER TABLE documents ADD COLUMN publication_year INTEGER;

-- Indexes for common filter/sort queries on book metadata
CREATE INDEX idx_documents_title ON documents(title);
CREATE INDEX idx_documents_author ON documents(author);
CREATE INDEX idx_documents_publication_year ON documents(publication_year);
