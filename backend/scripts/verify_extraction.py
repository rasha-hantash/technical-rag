#!/usr/bin/env python3
"""Verification script for PDF extraction quality.

Usage:
    python scripts/verify_extraction.py <pdf_path> [--pages N]

Outputs parsed content for manual verification of extraction quality.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_llm_server.rag.ingestion.pdf_parser import parse_pdf


def main():
    parser = argparse.ArgumentParser(description="Verify PDF extraction quality")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument(
        "--pages", type=int, default=5, help="Number of pages to display (default: 5)"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Parsing: {pdf_path}")
    print("=" * 80)

    doc = parse_pdf(pdf_path)

    print(f"Total pages: {doc.total_pages}")
    print(f"Showing first {min(args.pages, doc.total_pages)} pages")
    print("=" * 80)

    for page in doc.pages[: args.pages]:
        print(f"\n--- Page {page.page_number} ---")
        print(f"Blocks: {len(page.blocks)}, Tables: {len(page.tables)}")
        print()

        for block in page.blocks:
            type_marker = {
                "heading": "[H]",
                "paragraph": "[P]",
                "list_item": "[L]",
            }.get(block.block_type, "[?]")

            # Truncate long text for display
            text = block.text[:200] + "..." if len(block.text) > 200 else block.text
            print(f"  {type_marker} (size={block.font_size:.1f}) {text}")

        if page.tables:
            print("\n  Tables:")
            for table in page.tables:
                print(f"    Table {table.table_index}: {len(table.headers)} cols, {len(table.rows)} rows")
                if table.headers:
                    print(f"      Headers: {table.headers[:5]}{'...' if len(table.headers) > 5 else ''}")

    print("\n" + "=" * 80)
    print("Extraction complete.")


if __name__ == "__main__":
    main()
