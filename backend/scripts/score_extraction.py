#!/usr/bin/env python3
"""Score PDF extraction quality against ground truth annotations.

Usage:
    uv run python scripts/score_extraction.py <pdf_path> <ground_truth.json> [options]

    --json                      Output as JSON instead of text table
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from technical_rag.rag.ingestion.pdf_parser import parse_pdf_pymupdf

from eval.ground_truth import load_ground_truth
from eval.scoring import score_extraction, format_report, format_report_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score PDF extraction quality against ground truth annotations."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to parse")
    parser.add_argument(
        "ground_truth_path", type=Path, help="Path to the ground truth JSON file"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Output as JSON"
    )

    args = parser.parse_args()

    # Validate input files
    if not args.pdf_path.exists():
        print(f"Error: PDF file not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)
    if not args.ground_truth_path.exists():
        print(
            f"Error: Ground truth file not found: {args.ground_truth_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load ground truth
    gt = load_ground_truth(args.ground_truth_path)

    if not gt.pages:
        print("Warning: ground truth has no annotated pages.", file=sys.stderr)
        sys.exit(0)

    # Parse PDF with PyMuPDF
    doc = parse_pdf_pymupdf(args.pdf_path)

    # Score
    report = score_extraction(gt, doc, "pymupdf")

    # Output
    if args.json_output:
        print(format_report_json(report))
    else:
        print(format_report(report))


if __name__ == "__main__":
    main()
