#!/usr/bin/env python3
"""Generate an interactive HTML report for PDF extraction evaluation.

Usage:
    uv run python scripts/generate_report.py <pdf_path> [options]

    --pages N                   Limit to first N pages
    --dpi N                     Rendering DPI (default: 150)
    -o PATH                     Output path (default: <pdf_name>.report.html)
"""

import argparse
import hashlib
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from technical_rag.rag.ingestion.chunking import chunk_parsed_document
from technical_rag.rag.ingestion.pdf_parser import parse_pdf_pymupdf

from eval.report_renderer import generate_html_report


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML report for PDF extraction evaluation."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Limit to first N pages",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Rendering DPI (default: 150)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <pdf_name>.report.html)",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    if args.pages is not None and args.pages <= 0:
        parser.error("--pages must be a positive integer")

    # Compute file hash
    file_hash = compute_file_hash(args.pdf_path)

    # Parse with PyMuPDF
    print("Parsing with pymupdf...", file=sys.stderr)
    pymupdf_doc = parse_pdf_pymupdf(args.pdf_path)
    if args.pages is not None:
        pymupdf_doc.pages = pymupdf_doc.pages[: args.pages]
        pymupdf_doc.total_pages = len(pymupdf_doc.pages)
    pymupdf_chunks = chunk_parsed_document(pymupdf_doc)
    results = {"pymupdf": {"doc": pymupdf_doc, "chunks": pymupdf_chunks}}

    # Generate report
    print("Generating HTML report...", file=sys.stderr)
    html = generate_html_report(args.pdf_path, results, file_hash, dpi=args.dpi)

    # Write output
    output_path = args.output
    if output_path is None:
        output_path = args.pdf_path.with_suffix(".report.html")

    output_path.write_text(html)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
