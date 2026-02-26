ALTER TABLE chunks ADD COLUMN bbox JSONB;

COMMENT ON COLUMN chunks.bbox IS 'Bounding box coordinates on the page as JSON {x0, y0, x1, y1}';
