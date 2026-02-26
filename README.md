# Technical Book Rack

A RAG system for querying your personal library of technical books. Upload PDFs, ask questions, and get answers grounded in your books with full citations — book title, chapter, section, and page number.

## Core Use Case

Given code you've written, justify alignment with best practices from your technical books. The system also flags potentially outdated advice based on publication year.

```
You: "Is this Rust error handling pattern aligned with best practices?"
     [pastes code]

System: "According to [The Rust Programming Language, Ch. 9 > 9.2 > Recoverable Errors, p.142],
         using the ? operator for propagation is the recommended approach..."
```

## Architecture

```
Frontend (React/TS, TanStack Router)
   ↕ API
Backend (FastAPI, Python)
   ├─ PDF Parser (PyMuPDF + OCR fallback)
   │    ├─ Code block detection (monospace fonts)
   │    ├─ Heading hierarchy (font-size + TOC bookmarks)
   │    └─ Callout detection (TIP/WARNING/NOTE patterns)
   ├─ Concept-aware chunking
   │    ├─ Respects section boundaries
   │    ├─ Keeps code with surrounding explanation
   │    └─ Tracks section hierarchy per chunk
   ├─ Embeddings (OpenAI text-embedding-3-large, 3072D)
   ├─ Hybrid retrieval (cosine + BM25 + RRF)
   ├─ Reranking (Cohere rerank-v3.5)
   └─ Generation (Claude with book-aware citations)
   ↕
PostgreSQL 16 + pgvector
```

## Quick Start

### Prerequisites

- Python 3.14+
- Node.js 20+
- Docker (for PostgreSQL + pgvector)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [golang-migrate](https://github.com/golang-migrate/migrate) (database migrations)

### Setup

```bash
# Start PostgreSQL with pgvector
docker compose up -d

# Run database migrations
task db:migrate:up

# Install backend dependencies
cd backend && uv sync

# Install frontend dependencies
cd frontend && npm install
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Optional (recommended)
export COHERE_API_KEY=...        # For reranking
export RERANKER=cohere           # Enable Cohere reranking
```

### Run

```bash
# Backend (from backend/)
uv run python main.py

# Frontend (from frontend/)
npm run dev
```

## API

| Method | Endpoint                          | Description                      |
| ------ | --------------------------------- | -------------------------------- |
| POST   | `/api/v1/rag/ingest/batch`        | Upload PDFs with book metadata   |
| POST   | `/api/v1/rag/query`               | Ask a question, get cited answer |
| GET    | `/api/v1/documents`               | List all books                   |
| PUT    | `/api/v1/documents/{id}/metadata` | Update book metadata             |
| GET    | `/api/v1/documents/{id}/sections` | Get chapter/section tree         |
| GET    | `/api/v1/documents/{id}/file`     | Download original PDF            |

## Key Design Decisions

**PyMuPDF over Reducto** — Technical books are born-digital PDFs with clean text layers. PyMuPDF extracts directly from the PDF structure without API costs. Tesseract OCR is kept as a passive fallback for rare scanned books.

**Concept-aware chunking** — A "best practice" in a technical book is typically: explanation, code example, further explanation. The chunker keeps these together as one unit (max 2500 chars), never crossing section boundaries. Callouts (TIP/WARNING/NOTE) become their own chunks.

**text-embedding-3-large (3072D)** — Technical content has nuanced distinctions (e.g., "ownership" in Rust vs. general "ownership"). The higher fidelity is worth the minimal cost increase for a personal tool.

**Hybrid retrieval + reranking** — BM25 catches exact function names and patterns that embeddings miss. Cohere reranking sharpens results when searching across multiple 400-page books.

**TOC bookmark extraction** — `doc.get_toc()` gives reliable section hierarchy from PDF bookmarks, which most technical publishers include. Font-size heuristics serve as a fallback.

## Project Structure

```
backend/
  src/technical_rag/
    rag/
      ingestion/
        pdf_parser.py      # PyMuPDF + code/heading/callout detection
        chunking.py         # Concept-aware chunking algorithm
        pipeline.py         # End-to-end ingestion orchestration
      retrieval/
        retriever.py        # Hybrid search (cosine + BM25 + RRF)
        reranker.py         # Cohere / cross-encoder reranking
      generation/
        generator.py        # Claude with book-citation prompt
      database.py           # pgvector operations
      models.py             # Pydantic models
    server.py               # FastAPI endpoints
  migrations/               # PostgreSQL schema migrations

frontend/
  src/
    components/
      ChatPanel.tsx         # Main chat interface
      EvidencePanel.tsx     # Source citations sidebar
      EvidenceCard.tsx      # Individual source with book metadata
      DocumentList.tsx      # Book shelf display
      PdfPageViewer.tsx     # PDF page with bbox highlighting
    hooks/                  # React hooks for query, ingest, documents
    lib/                    # API client and TypeScript types

docs/                       # Your PDF books go here
```

## Forked From

This project is forked from [pdf-classaction-rag](https://github.com/rasha-hantash/pdf-classaction-rag), a RAG system for legal documents. Adapted for technical books with code-aware parsing, concept-based chunking, and book citation support.
