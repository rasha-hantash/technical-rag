"""Retrieval evaluation: gold passage matching and retrieval metrics.

Matches gold passages from the evaluation dataset against retrieved chunks
using substring and fuzzy matching to compute Recall@k, Precision@k, and MRR.
"""

import difflib
from dataclasses import dataclass, field

from .rag_ground_truth import GoldPassage

_FUZZY_THRESHOLD = 0.8


def passage_in_chunk(passage_text: str, chunk_text: str) -> bool:
    """Check if a gold passage is contained in a retrieved chunk.

    First tries substring match (fast). Falls back to fuzzy matching
    with SequenceMatcher >= 0.8 threshold for passages that are close
    but not exact (e.g., minor whitespace or formatting differences).
    """
    passage_norm = " ".join(passage_text.split()).lower()
    chunk_norm = " ".join(chunk_text.split()).lower()

    if not passage_norm or not chunk_norm:
        return False

    # Fast path: exact substring
    if passage_norm in chunk_norm:
        return True

    # Fuzzy match: slide a window of passage length over the chunk
    # and check if any window exceeds the threshold
    if len(passage_norm) == 0:
        return False

    if len(chunk_norm) < len(passage_norm):
        ratio = difflib.SequenceMatcher(None, passage_norm, chunk_norm).ratio()
        return ratio >= _FUZZY_THRESHOLD

    window_size = len(passage_norm)
    best_ratio = 0.0
    # Step by word boundaries for efficiency
    words = chunk_norm.split()
    chunk_positions = []
    pos = 0
    for word in words:
        idx = chunk_norm.index(word, pos)
        chunk_positions.append(idx)
        pos = idx + len(word)

    for start_pos in chunk_positions:
        window = chunk_norm[start_pos : start_pos + window_size]
        if not window:
            continue
        ratio = difflib.SequenceMatcher(None, passage_norm, window).ratio()
        if ratio >= _FUZZY_THRESHOLD:
            return True
        best_ratio = max(best_ratio, ratio)

    return False


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics for a single question."""

    question: str = ""
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    gold_count: int = 0
    retrieved_count: int = 0
    matched_count: int = 0


def compute_retrieval_metrics(
    gold_passages: list[GoldPassage],
    retrieved_texts: list[str],
) -> RetrievalMetrics:
    """Compute retrieval metrics for one question.

    Args:
        gold_passages: Expected passages from the evaluation dataset.
        retrieved_texts: Text content of retrieved chunks, in rank order.

    Returns:
        RetrievalMetrics with recall@k, precision@k, and MRR.
    """
    metrics = RetrievalMetrics(
        gold_count=len(gold_passages),
        retrieved_count=len(retrieved_texts),
    )

    if not gold_passages or not retrieved_texts:
        return metrics

    # Track which gold passages have been found
    found_golds: set[int] = set()
    first_hit_rank: int | None = None

    for rank, chunk_text in enumerate(retrieved_texts):
        for gi, gp in enumerate(gold_passages):
            if gi in found_golds:
                continue
            if passage_in_chunk(gp.text, chunk_text):
                found_golds.add(gi)
                if first_hit_rank is None:
                    first_hit_rank = rank

    metrics.matched_count = len(found_golds)

    # Recall@k: fraction of gold passages found in top-k retrieved
    metrics.recall_at_k = len(found_golds) / len(gold_passages)

    # Precision@k: fraction of retrieved chunks that contain a gold passage
    hits = 0
    for chunk_text in retrieved_texts:
        for gp in gold_passages:
            if passage_in_chunk(gp.text, chunk_text):
                hits += 1
                break
    metrics.precision_at_k = hits / len(retrieved_texts) if retrieved_texts else 0.0

    # MRR: reciprocal of the rank of the first relevant result (1-indexed)
    if first_hit_rank is not None:
        metrics.reciprocal_rank = 1.0 / (first_hit_rank + 1)

    return metrics


@dataclass
class AggregateRetrievalMetrics:
    """Aggregated retrieval metrics across all questions."""

    avg_recall_at_k: float = 0.0
    avg_precision_at_k: float = 0.0
    avg_mrr: float = 0.0
    total_questions: int = 0
    per_question: list[RetrievalMetrics] = field(default_factory=list)


def aggregate_retrieval_metrics(
    per_question: list[RetrievalMetrics],
) -> AggregateRetrievalMetrics:
    """Aggregate retrieval metrics across multiple questions."""
    if not per_question:
        return AggregateRetrievalMetrics()

    n = len(per_question)
    return AggregateRetrievalMetrics(
        avg_recall_at_k=sum(m.recall_at_k for m in per_question) / n,
        avg_precision_at_k=sum(m.precision_at_k for m in per_question) / n,
        avg_mrr=sum(m.reciprocal_rank for m in per_question) / n,
        total_questions=n,
        per_question=per_question,
    )
