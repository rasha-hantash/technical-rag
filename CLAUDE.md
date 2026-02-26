# Technical Book Rack

RAG system for querying personal technical book PDFs. FastAPI backend + React/TypeScript frontend.

## Stack

- **Backend:** Python 3.14+, FastAPI, PyMuPDF, pgvector, OpenAI embeddings (text-embedding-3-large, 3072D), Cohere reranking, Claude generation
- **Frontend:** React, TypeScript, TanStack Router, Vite
- **Database:** PostgreSQL 16 + pgvector
- **Package managers:** uv (backend), npm (frontend)

## Key design decisions — do not change without discussion

- **Concept-aware chunking** keeps code + explanation together as one unit (max 2500 chars). Never crosses section boundaries. Callouts (TIP/WARNING/NOTE) become their own chunks.
- **Section hierarchy** is tracked per chunk (e.g., "Chapter 5 > 5.3 > CQRS Pattern") for citation purposes.
- **Book metadata** (title, author, publication_year) is threaded through the entire pipeline — ingestion, storage, retrieval, generation, and frontend display.
- **text-embedding-3-large (3072D)** was chosen over text-embedding-3-small for higher fidelity on nuanced technical concepts.
- **PyMuPDF only** (no Reducto) — technical books are clean born-digital PDFs. Tesseract OCR is a passive fallback only.
- **Hybrid retrieval** (cosine + BM25 + RRF) + Cohere reranking catches both semantic and exact-match queries across multiple 400-page books.

## Running

```bash
# Backend
cd backend && uv run python main.py

# Frontend
cd frontend && npm run dev

# Tests
cd backend && uv run pytest
cd frontend && npx playwright test

# Database migrations
task db:migrate:up
```
