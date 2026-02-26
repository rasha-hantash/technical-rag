"""CLI entry point for RAG pipeline matrix evaluation.

Usage:
    cd backend && uv run python scripts/run_rag_eval.py datasets/classaction-v1.json --variants quick --skip-generation
    cd backend && uv run python scripts/run_rag_eval.py datasets/classaction-v1.json --variants quick
    cd backend && uv run python scripts/run_rag_eval.py datasets/classaction-v1.json --variants all --json
"""

import argparse
import sys
from pathlib import Path

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add scripts/ to path for eval package imports
sys.path.insert(0, str(Path(__file__).parent))

from eval.matrix_runner import (
    format_results_json,
    format_results_table,
    run_matrix,
)
from eval.pipeline_config import get_default_matrix, get_quick_matrix
from eval.rag_ground_truth import load_eval_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG pipeline matrix evaluation"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--variants",
        choices=["quick", "all"],
        default="quick",
        help="Variant set to run: 'quick' (2 variants) or 'all' (12 variants)",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip LLM judge evaluation (retrieval-only, cheaper)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON instead of table",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Claude model to use as generation judge",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="PostgreSQL URL for eval database (default: EVAL_DATABASE_URL env var)",
    )

    args = parser.parse_args()

    # Resolve dataset path relative to scripts/eval/ if not absolute
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        # Try relative to scripts/eval/ first (for datasets/ subdir)
        eval_dir = Path(__file__).parent / "eval"
        candidate = eval_dir / dataset_path
        if candidate.exists():
            dataset_path = candidate
        else:
            # Try relative to CWD
            dataset_path = Path.cwd() / args.dataset

    dataset = load_eval_dataset(dataset_path)
    print(
        f"Loaded dataset: {dataset.name or dataset_path.name} "
        f"({len(dataset.questions)} questions, {len(dataset.pdf_corpus)} PDFs)"
    )

    if args.variants == "quick":
        variants = get_quick_matrix()
    else:
        variants = get_default_matrix()

    print(f"Running {len(variants)} variant(s)...")
    if args.skip_generation:
        print("Generation eval: SKIPPED")
    print()

    results = run_matrix(
        variants=variants,
        dataset=dataset,
        connection_string=args.db_url,
        skip_generation=args.skip_generation,
        top_k=args.top_k,
        judge_model=args.judge_model,
    )

    if args.json_output:
        print(format_results_json(results))
    else:
        print(format_results_table(results))


if __name__ == "__main__":
    main()
