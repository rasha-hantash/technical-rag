#!/usr/bin/env bash
# Downloads the Google Cloud "Startup Technical Guide: AI Agents" PDF
# and ingests it into the RAG system.
#
# Usage:
#   ./scripts/download_google_agents_guide.sh          # download only
#   ./scripts/download_google_agents_guide.sh --ingest # download + ingest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$PROJECT_ROOT/docs"
PDF_URL="https://services.google.com/fh/files/misc/startup_technical_guide_ai_agents_final.pdf"
PDF_FILE="$DOCS_DIR/startup_technical_guide_ai_agents.pdf"

mkdir -p "$DOCS_DIR"

if [ -f "$PDF_FILE" ]; then
    echo "PDF already exists at $PDF_FILE"
else
    echo "Downloading PDF..."
    curl -L -o "$PDF_FILE" "$PDF_URL"
    echo "Downloaded to $PDF_FILE"
fi

if [ "${1:-}" = "--ingest" ]; then
    echo "Ingesting into RAG system..."
    curl -X POST http://localhost:8000/api/v1/rag/ingest/batch \
        -F "files=@$PDF_FILE" \
        -F "title=Startup Technical Guide: AI Agents" \
        -F "author=Google Cloud" \
        -F "publication_year=2025" \
        -F 'tags=["ai-agents", "google-cloud", "startups"]'
    echo ""
    echo "Ingestion complete!"
fi
