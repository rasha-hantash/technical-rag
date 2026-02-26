# Technical Book Rack RAG — Implementation Plan

## Context

Build a personal "bookshelf" RAG system for technical books (200-550 page PDFs) — primarily architecture and design pattern books. The core use case: given code you've written, justify alignment with best practices from books, with full source citations (book title, chapter, page). The system should also flag potentially outdated advice based on publication year.

**Starting point:** Fork `pdf-classaction-rag` (`/Users/rashasaadeh/workspace/personal/pdf-classaction-rag`), which already has a production-quality RAG pipeline (PyMuPDF parsing, pgvector hybrid search, Cohere reranking, Claude generation, React frontend with bbox highlighting).

**Key decisions:**

- PyMuPDF only (no Reducto) — technical books are clean PDFs. Keep existing Tesseract OCR as passive fallback for rare scanned books.
- `text-embedding-3-large` (3072D) — higher fidelity for nuanced technical concepts
- Hybrid retrieval (cosine + BM25 + RRF) + Cohere reranking from day one
- Concept-aware semantic chunking (not fixed-size)

---

## Phase 0: Fork and Rename

Copy `pdf-classaction-rag` → `technical-rag`. Rename package `pdf_llm_server` → `technical_rag`.

**Files to change:**

- `backend/pyproject.toml` — name, package path
- `backend/main.py` — import path
- `backend/src/pdf_llm_server/` → `backend/src/technical_rag/` (directory rename)
- `docker-compose.yaml` — container name, db name
- `Taskfile.yml` — database URL
- `frontend/package.json` — name
- `frontend/src/routes/__root.tsx` — title

**Also:** Remove Reducto parser:

- Delete `reducto_parser.py`
- Remove `reducto_parser` param from `parse_pdf()`, `ingest_document()`, `RAGIngestionPipeline`, `server.py` lifespan
- Remove `reductoai` from `pyproject.toml`

---

## Phase 1: Schema Migrations

### Migration 000006 — Book metadata on documents

```sql
ALTER TABLE documents ADD COLUMN title TEXT;
ALTER TABLE documents ADD COLUMN author TEXT;
ALTER TABLE documents ADD COLUMN edition VARCHAR(50);
ALTER TABLE documents ADD COLUMN publication_year INTEGER;
```

### Migration 000007 — Section hierarchy + 3072D embeddings

```sql
ALTER TABLE chunks ADD COLUMN section_hierarchy TEXT;
ALTER TABLE chunks DROP COLUMN embedding;
ALTER TABLE chunks ADD COLUMN embedding vector(3072);
```

Fresh fork = no existing data to preserve, so drop+recreate is fine.

---

## Phase 2: Models

### `parser_models.py` — Add to `TextBlock`:

- `is_monospace: bool = False` — for code block detection
- `heading_level: int | None = None` — 1=chapter, 2=section, 3=subsection
- `callout_type: str | None = None` — "tip", "warning", "note", "best_practice"

### `models.py` — Add to `IngestedDocument`:

- `title`, `author`, `edition`, `publication_year`

### `models.py` — Add to `ChunkData` and `ChunkRecord`:

- `section_hierarchy: str | None = None`

---

## Phase 3: PDF Parser Enhancements

**File:** `backend/src/technical_rag/rag/ingestion/pdf_parser.py`

### 3a. TOC/bookmark extraction (primary source of section hierarchy)

Use `doc.get_toc()` which returns `[[level, title, page], ...]`. Build a page→heading mapping that the chunker uses to assign `section_hierarchy`. This is more reliable than font heuristics for most technical books.

```python
def _extract_toc(doc: fitz.Document) -> dict[int, list[tuple[int, str]]]:
    """Extract TOC bookmarks, returning {page_number: [(level, title), ...]}."""
    toc = doc.get_toc()  # [[level, title, page], ...]
    # Group by page number for lookup during chunking
```

### 3b. Monospace font detection in `_extract_spans_info`

Return 4-tuple: `(text, avg_font_size, is_bold, is_monospace)`. Detect monospace via font name matching: `courier`, `mono`, `consolas`, `menlo`, `source code`, `fira code`, `inconsolata`.

Require >50% monospace spans AND >20 chars to classify as `code_block` (avoids false positives from inline code).

### 3c. Heading level assignment in `_classify_block`

Return `(block_type, heading_level)`:

- `font_size > median * 1.5` → heading level 1
- `font_size > median * 1.2` → heading level 2
- Bold + short text → heading level 3

### 3d. Callout detection

Pattern-match block text for prefixes: `TIP:`, `WARNING:`, `NOTE:`, `BEST PRACTICE:`, `DEFINITION:`. Set `block_type = "callout"` and `callout_type` accordingly.

### 3e. Wire into `parse_pdf_pymupdf`

- First pass: collect font sizes (existing) + extract TOC
- Second pass: use updated `_extract_spans_info` and `_classify_block`, populate new `TextBlock` fields
- Attach TOC data to `ParsedDocument` (add `toc: dict` field)

---

## Phase 4: Concept-Aware Chunking

**File:** `backend/src/technical_rag/rag/ingestion/chunking.py`

New function `concept_aware_chunking(doc, max_chunk_size=2500)` — default strategy.

**Algorithm:**

1. Walk all pages, maintain a `heading_stack: list[str]` (indexed by level)
2. On heading block: flush accumulator, update heading stack, compute `section_hierarchy = " > ".join(heading_stack)`. If TOC data available, prefer TOC title over detected heading text.
3. On callout block: flush, emit as own chunk with `chunk_type = callout_type`
4. On code block: do NOT flush — keep with preceding explanation. Only flush if adding would exceed `max_chunk_size`
5. On paragraph/list: accumulate, flush when exceeding `max_chunk_size`
6. Tables: separate chunks (existing behavior)

**Key principle:** A "best practice" in a technical book = explanation + code example + further explanation. This must be one chunk. `max_chunk_size=2500` (up from 1500) gives room for architecture concepts.

Reuse existing `_find_chunk_block_bboxes` for bbox tracking.

---

## Phase 5: Embeddings Update

**File:** `backend/src/technical_rag/rag/llm_clients/embeddings.py`

Change constants:

- `MODEL = "text-embedding-3-large"` (was `text-embedding-3-small`)
- `EMBEDDING_DIMENSIONS = 3072` (was `1536`)

Same tokenizer (`cl100k_base`), same batch logic — no other changes needed.

---

## Phase 6: Database Layer

**File:** `backend/src/technical_rag/rag/database.py`

- `insert_document` — accept `title`, `author`, `edition`, `publication_year` params
- `insert_chunks` — include `section_hierarchy` in INSERT
- `_rows_to_search_results` — hydrate `section_hierarchy` on ChunkRecord, book metadata on IngestedDocument
- `get_documents` — SELECT new columns
- Search queries — add `c.section_hierarchy` to SELECT
- **New:** `get_book_sections(document_id)` — distinct section hierarchies + chunk counts for browsing
- **New:** `update_book_metadata(document_id, ...)` — update title/author/edition/year

---

## Phase 7: Ingestion Pipeline

**File:** `backend/src/technical_rag/rag/ingestion/pipeline.py`

- `ingest_document` — accept book metadata params, default `chunking_strategy="concept"`
- Pass metadata through to `db.insert_document()`
- Include `section_hierarchy` when constructing `ChunkRecord` from `ChunkData`

---

## Phase 8: Generation Prompt

**File:** `backend/src/technical_rag/rag/generation/generator.py`

### New system prompt:

```
You are a technical architecture advisor that answers questions by referencing authoritative technical books.

Rules:
1. Only use information from the provided context.
2. Cite sources as: [BookTitle, Chapter/Section, p.XX]
3. If multiple books discuss the same topic, compare their perspectives.
4. If a source's publication year suggests outdated advice, note it explicitly.
5. When evaluating code, be specific about which principle applies and why.
6. If the context is insufficient, say so clearly.
```

### Updated `_build_context`:

Include book title + author + year, section hierarchy, and chunk type in each context header.

### Updated `SourceReference`:

Add `section_hierarchy`, `book_title`, `book_author`, `publication_year`.

---

## Phase 9: Backend API

**File:** `backend/src/technical_rag/server.py`

- Update ingest endpoint to accept book metadata as form fields (title, author, edition, publication_year)
- Update `DocumentResponse` and `SourceResponse` with new fields
- **New:** `PUT /api/v1/documents/{id}/metadata` — update book metadata post-upload
- **New:** `GET /api/v1/documents/{id}/sections` — get chapter/section tree
- Increase `MAX_UPLOAD_SIZE` to 200MB for large technical books

---

## Phase 10: Frontend

### Types + API (`lib/types.ts`, `lib/api.ts`)

- Add book metadata fields to `DocumentResponse`, `SourceResponse`
- New `BookMetadata`, `BookSection` interfaces
- Update `ingestBatch` to accept optional metadata
- New `updateBookMetadata`, `getBookSections` API functions

### EvidenceCard update (`components/EvidenceCard.tsx`)

- Show book title instead of filename
- Show section hierarchy path
- Show publication year badge

### New: BookShelf component (replaces `DocumentList`)

- Book cards with title/author/year
- Upload button that opens metadata form

### New: BookUploadDialog component

- File picker + text inputs for title, author, edition, year

---

## Stack Structure (for `gt`)

| Diff | Content                                                 | Phases |
| ---- | ------------------------------------------------------- | ------ |
| 1    | Fork rename + Reducto removal                           | 0      |
| 2    | Schema migrations                                       | 1      |
| 3    | Models + parser enhancements (TOC, monospace, callouts) | 2-3    |
| 4    | Concept-aware chunking + embeddings update              | 4-5    |
| 5    | Database layer + pipeline + generation prompt           | 6-8    |
| 6    | Backend API endpoints                                   | 9      |
| 7    | Frontend (types, API, BookShelf, EvidenceCard)          | 10     |

---

## Verification

1. **Parser:** Ingest a real technical book PDF. Verify code blocks are detected (monospace), headings have levels, TOC is extracted. Print parsed blocks and spot-check.
2. **Chunking:** Verify chunks respect section boundaries, code stays with explanation, callouts are separate. Check `section_hierarchy` is populated.
3. **End-to-end:** Upload a book with metadata → query with a code snippet → verify response cites book title + chapter + page, and EvidenceCard displays it.
4. **Outdated detection:** Upload an old book (pre-2018) → ask about a topic with evolving best practices → verify Claude flags the publication year.
5. **Hybrid search:** Query with both natural language ("what's the CQRS pattern?") and code (`fn handle_command(...)`) → verify both return relevant results.

---

## Future (not in scope)

- Auto-extract book metadata from title/copyright pages (via heuristics or Claude call)
- Document-scoped search (`document_ids` filter on query endpoint)
- Dedicated `/library` route for full-page bookshelf browsing
- HNSW index for 3072D vectors (once >1000 chunks)
- Staleness scoring model (topic → knowledge half-life)
