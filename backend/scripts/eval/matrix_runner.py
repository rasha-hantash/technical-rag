"""Matrix evaluation runner: loops pipeline variants and collects results.

For each variant, truncates the eval database, ingests the PDF corpus,
runs retrieval and generation evaluation, and collects per-stage metrics.
"""

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from pdf_llm_server.logger import logger
from pdf_llm_server.rag.database import PgVectorStore
from pdf_llm_server.rag.ingestion.pipeline import ingest_document
from pdf_llm_server.rag.ingestion.reducto_parser import ReductoParser
from pdf_llm_server.rag.llm_clients.anthropic_client import AnthropicClient
from pdf_llm_server.rag.llm_clients.embeddings import EmbeddingClient
from pdf_llm_server.rag.models import SearchResult
from pdf_llm_server.rag.retrieval.reranker import (
    CohereReranker,
    CrossEncoderReranker,
    Reranker,
)
from pdf_llm_server.rag.retrieval.retriever import RAGRetriever
from pdf_llm_server.rag.generation.generator import RAGGenerator

from .generation_eval import (
    AggregateGenerationMetrics,
    aggregate_generation_metrics,
    judge_generation,
)
from .pipeline_config import PipelineVariant
from .rag_ground_truth import EvalDataset
from .retrieval_eval import (
    AggregateRetrievalMetrics,
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
)


@dataclass
class ChunkingDiagnostics:
    """Diagnostic stats about chunks produced during ingestion."""

    chunk_count: int = 0
    avg_size: float = 0.0
    median_size: float = 0.0
    min_size: int = 0
    max_size: int = 0
    size_std: float = 0.0


@dataclass
class VariantResult:
    """Evaluation results for a single pipeline variant."""

    variant: PipelineVariant | None = None
    chunking: ChunkingDiagnostics | None = None
    retrieval: AggregateRetrievalMetrics | None = None
    generation: AggregateGenerationMetrics | None = None
    ingestion_duration_ms: float = 0.0
    retrieval_duration_ms: float = 0.0
    generation_duration_ms: float = 0.0
    error: str | None = None


def _build_reranker(reranker_name: str) -> Reranker | None:
    if reranker_name == "none":
        return None
    if reranker_name == "cohere":
        return CohereReranker()
    if reranker_name == "cross-encoder":
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker: {reranker_name}")


def _compute_chunking_diagnostics(db: PgVectorStore) -> ChunkingDiagnostics:
    """Query chunk sizes from the database and compute diagnostics."""
    with db.conn.cursor() as cur:
        cur.execute("SELECT LENGTH(content) FROM chunks")
        sizes = [row[0] for row in cur.fetchall()]

    if not sizes:
        return ChunkingDiagnostics()

    sizes_sorted = sorted(sizes)
    n = len(sizes_sorted)
    mean = sum(sizes_sorted) / n
    median = (
        sizes_sorted[n // 2]
        if n % 2 == 1
        else (sizes_sorted[n // 2 - 1] + sizes_sorted[n // 2]) / 2
    )
    variance = sum((s - mean) ** 2 for s in sizes_sorted) / n
    std = variance**0.5

    return ChunkingDiagnostics(
        chunk_count=n,
        avg_size=round(mean, 1),
        median_size=round(median, 1),
        min_size=sizes_sorted[0],
        max_size=sizes_sorted[-1],
        size_std=round(std, 1),
    )


def _set_parser_env(parser: str) -> str | None:
    """Set PDF_PARSER env var and return previous value for restore."""
    prev = os.environ.get("PDF_PARSER")
    os.environ["PDF_PARSER"] = parser
    return prev


def _restore_parser_env(prev: str | None) -> None:
    if prev is None:
        os.environ.pop("PDF_PARSER", None)
    else:
        os.environ["PDF_PARSER"] = prev


def run_variant(
    variant: PipelineVariant,
    dataset: EvalDataset,
    db: PgVectorStore,
    embedding_client: EmbeddingClient,
    reducto_parser: ReductoParser | None,
    judge_client: AnthropicClient | None,
    skip_generation: bool = False,
    top_k: int = 5,
) -> VariantResult:
    """Run evaluation for a single pipeline variant.

    Steps:
    1. Truncate all tables
    2. Ingest corpus with variant's parser + chunking
    3. Run retrieval eval for each question
    4. Optionally run generation eval for each question
    """
    result = VariantResult(variant=variant)

    logger.info(
        "starting variant evaluation",
        variant=variant.name,
        parser=variant.parser,
        chunking=variant.chunking,
        reranker=variant.reranker,
    )

    # Step 1: clean slate
    db.truncate_tables()

    # Step 2: ingest corpus
    prev_parser = _set_parser_env(variant.parser)
    try:
        rp = reducto_parser if variant.parser == "reducto" else None
        ingest_start = time.perf_counter()

        for pdf_path in dataset.pdf_corpus:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                logger.error("pdf not found, skipping", pdf_path=str(pdf_file))
                continue

            ingest_result = ingest_document(
                file_path=pdf_file,
                db=db,
                embedding_client=embedding_client,
                chunking_strategy=variant.chunking,
                reducto_parser=rp,
            )
            if ingest_result.error:
                logger.error(
                    "ingestion failed",
                    pdf_path=str(pdf_file),
                    error=ingest_result.error,
                )

        result.ingestion_duration_ms = (time.perf_counter() - ingest_start) * 1000
    finally:
        _restore_parser_env(prev_parser)

    # Chunking diagnostics
    result.chunking = _compute_chunking_diagnostics(db)
    logger.info(
        "ingestion complete",
        variant=variant.name,
        chunk_count=result.chunking.chunk_count,
        avg_chunk_size=result.chunking.avg_size,
        duration_ms=round(result.ingestion_duration_ms, 2),
    )

    # Step 3: retrieval eval
    reranker = _build_reranker(variant.reranker)
    retriever = RAGRetriever(
        db=db, embedding_client=embedding_client, reranker=reranker
    )

    retrieval_start = time.perf_counter()
    retrieval_per_question = []

    for eq in dataset.questions:
        results: list[SearchResult] = retriever.retrieve(eq.question, top_k=top_k)
        retrieved_texts = [r.chunk.content for r in results]

        rm = compute_retrieval_metrics(eq.gold_passages, retrieved_texts)
        rm.question = eq.question
        retrieval_per_question.append(rm)

    result.retrieval = aggregate_retrieval_metrics(retrieval_per_question)
    result.retrieval_duration_ms = (time.perf_counter() - retrieval_start) * 1000

    logger.info(
        "retrieval eval complete",
        variant=variant.name,
        avg_recall=round(result.retrieval.avg_recall_at_k, 4),
        avg_mrr=round(result.retrieval.avg_mrr, 4),
        duration_ms=round(result.retrieval_duration_ms, 2),
    )

    # Step 4: generation eval (optional)
    if not skip_generation and judge_client is not None:
        generator = RAGGenerator(anthropic_client=judge_client)
        gen_start = time.perf_counter()
        gen_per_question = []

        for eq in dataset.questions:
            # Re-retrieve for generation (retriever is stateless)
            results = retriever.retrieve(eq.question, top_k=top_k)
            rag_response = generator.generate(eq.question, results)

            gm = judge_generation(
                question=eq.question,
                expected_answer=eq.expected_answer.answer_text,
                key_facts=eq.expected_answer.key_facts,
                generated_answer=rag_response.answer,
                judge_client=judge_client,
            )
            gen_per_question.append(gm)

        result.generation = aggregate_generation_metrics(gen_per_question)
        result.generation_duration_ms = (time.perf_counter() - gen_start) * 1000

        logger.info(
            "generation eval complete",
            variant=variant.name,
            avg_accuracy=round(result.generation.avg_factual_accuracy, 4),
            hallucination_rate=round(result.generation.hallucination_rate, 4),
            duration_ms=round(result.generation_duration_ms, 2),
        )

    return result


def run_matrix(
    variants: list[PipelineVariant],
    dataset: EvalDataset,
    connection_string: str | None = None,
    skip_generation: bool = False,
    top_k: int = 5,
    judge_model: str = "claude-haiku-4-5-20251001",
) -> list[VariantResult]:
    """Run the full evaluation matrix.

    Args:
        variants: List of pipeline variants to evaluate.
        dataset: Evaluation dataset with questions and corpus.
        connection_string: PostgreSQL URL for the eval database.
            Defaults to EVAL_DATABASE_URL env var.
        skip_generation: Skip LLM judge evaluation.
        top_k: Number of chunks to retrieve per question.
        judge_model: Claude model to use as judge.

    Returns:
        List of VariantResult, one per variant.
    """
    conn_str = connection_string or os.environ.get("EVAL_DATABASE_URL")
    if not conn_str:
        raise ValueError(
            "No eval database URL: set EVAL_DATABASE_URL or pass connection_string"
        )

    # Resolve PDF paths relative to dataset file (if relative)
    resolved_corpus = []
    for p in dataset.pdf_corpus:
        pp = Path(p)
        if not pp.is_absolute():
            raise ValueError(
                f"PDF path must be absolute: {p}. "
                "Use absolute paths in the dataset pdf_corpus."
            )
        resolved_corpus.append(str(pp))
    dataset.pdf_corpus = resolved_corpus

    embedding_client = EmbeddingClient()

    # Reducto parser (shared across variants that need it)
    reducto_parser = None
    needs_reducto = any(v.parser == "reducto" for v in variants)
    if needs_reducto:
        try:
            reducto_parser = ReductoParser()
        except (ValueError, ImportError) as e:
            logger.warn(
                "reducto parser not available, reducto variants will be skipped",
                error=str(e),
            )

    # Judge client for generation eval
    judge_client = None
    if not skip_generation:
        try:
            judge_client = AnthropicClient(model=judge_model)
        except ValueError as e:
            logger.warn(
                "anthropic client not available, generation eval will be skipped",
                error=str(e),
            )

    db = PgVectorStore(conn_str)
    db.connect()
    migrations_dir = Path(__file__).parent.parent.parent / "migrations"
    db.run_migrations(migrations_dir)

    results = []
    total = len(variants)

    try:
        for i, variant in enumerate(variants):
            logger.info(
                "running variant",
                variant=variant.name,
                progress=f"{i + 1}/{total}",
            )

            # Skip reducto variants if parser unavailable
            if variant.parser == "reducto" and reducto_parser is None:
                vr = VariantResult(variant=variant, error="reducto parser not available")
                results.append(vr)
                continue

            try:
                vr = run_variant(
                    variant=variant,
                    dataset=dataset,
                    db=db,
                    embedding_client=embedding_client,
                    reducto_parser=reducto_parser,
                    judge_client=judge_client,
                    skip_generation=skip_generation,
                    top_k=top_k,
                )
                results.append(vr)
            except Exception as e:
                logger.error(
                    "variant evaluation failed",
                    variant=variant.name,
                    error=str(e),
                )
                results.append(VariantResult(variant=variant, error=str(e)))
    finally:
        db.disconnect()

    return results


def format_results_table(results: list[VariantResult]) -> str:
    """Format results as a CLI comparison table."""
    header = (
        f"{'Variant':<35} | {'Chunks':>6} | {'Avg Size':>8} | "
        f"{'Recall@5':>8} | {'MRR':>6} | {'Accuracy':>8} | {'Halluc':>6}"
    )
    separator = "-" * len(header)

    lines = [
        "RAG Pipeline Matrix Evaluation Results",
        "=" * 38,
        "",
        header,
        separator,
    ]

    for vr in results:
        if vr.error:
            lines.append(f"{vr.variant.name:<35} | ERROR: {vr.error}")
            continue

        chunks = vr.chunking.chunk_count if vr.chunking else 0
        avg_size = vr.chunking.avg_size if vr.chunking else 0
        recall = vr.retrieval.avg_recall_at_k if vr.retrieval else 0
        mrr = vr.retrieval.avg_mrr if vr.retrieval else 0

        if vr.generation:
            accuracy = vr.generation.avg_factual_accuracy
            halluc_rate = vr.generation.hallucination_rate
            accuracy_str = f"{accuracy:.2f}"
            halluc_str = f"{halluc_rate * 100:.1f}%"
        else:
            accuracy_str = "   -"
            halluc_str = "  -"

        lines.append(
            f"{vr.variant.name:<35} | {chunks:>6} | {avg_size:>8.0f} | "
            f"{recall:>8.2f} | {mrr:>6.2f} | {accuracy_str:>8} | {halluc_str:>6}"
        )

    return "\n".join(lines)


def format_results_json(results: list[VariantResult]) -> str:
    """Format results as JSON for programmatic analysis."""
    data = []
    for vr in results:
        entry = {
            "variant": vr.variant.name if vr.variant else "unknown",
            "parser": vr.variant.parser if vr.variant else "",
            "chunking": vr.variant.chunking if vr.variant else "",
            "reranker": vr.variant.reranker if vr.variant else "",
            "error": vr.error,
        }

        if vr.chunking:
            entry["chunking_diagnostics"] = {
                "chunk_count": vr.chunking.chunk_count,
                "avg_size": vr.chunking.avg_size,
                "median_size": vr.chunking.median_size,
                "min_size": vr.chunking.min_size,
                "max_size": vr.chunking.max_size,
                "size_std": vr.chunking.size_std,
            }

        if vr.retrieval:
            entry["retrieval"] = {
                "avg_recall_at_k": round(vr.retrieval.avg_recall_at_k, 4),
                "avg_precision_at_k": round(vr.retrieval.avg_precision_at_k, 4),
                "avg_mrr": round(vr.retrieval.avg_mrr, 4),
                "total_questions": vr.retrieval.total_questions,
            }

        if vr.generation:
            entry["generation"] = {
                "avg_factual_accuracy": round(
                    vr.generation.avg_factual_accuracy, 4
                ),
                "avg_completeness": round(vr.generation.avg_completeness, 4),
                "avg_key_facts_recall": round(
                    vr.generation.avg_key_facts_recall, 4
                ),
                "hallucination_rate": round(vr.generation.hallucination_rate, 4),
                "total_questions": vr.generation.total_questions,
                "judge_errors": vr.generation.judge_errors,
            }

        entry["timing"] = {
            "ingestion_ms": round(vr.ingestion_duration_ms, 2),
            "retrieval_ms": round(vr.retrieval_duration_ms, 2),
            "generation_ms": round(vr.generation_duration_ms, 2),
        }

        data.append(entry)

    return json.dumps(data, indent=2)
