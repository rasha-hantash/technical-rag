"""Generation evaluation: LLM-as-judge for RAG answer quality.

Uses Claude as a judge to score generated answers against expected answers
on factual accuracy, completeness, key facts recall, and hallucination.
"""

import json
from dataclasses import dataclass

from technical_rag.rag.llm_clients.anthropic_client import AnthropicClient

_JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for a RAG (Retrieval-Augmented Generation) system.
You will be given a question, the expected answer with key facts, and the generated answer.
Score the generated answer on the following dimensions.

Respond ONLY with a JSON object (no markdown, no explanation) with these fields:
- "factual_accuracy": float 0.0-1.0 (how factually correct is the answer?)
- "completeness": float 0.0-1.0 (how much of the expected answer is covered?)
- "key_facts_found": list of strings (which key facts from the expected answer appear in the generated answer?)
- "key_facts_missing": list of strings (which key facts are missing?)
- "hallucinations": list of strings (claims in the generated answer not supported by the expected answer)
- "notes": string (brief explanation of scores)"""


_JUDGE_USER_TEMPLATE = """Question: {question}

Expected answer: {expected_answer}

Key facts that should be present: {key_facts}

Generated answer: {generated_answer}

Score the generated answer. Respond with JSON only."""


@dataclass
class GenerationMetrics:
    """Generation quality metrics for a single question."""

    question: str = ""
    factual_accuracy: float = 0.0
    completeness: float = 0.0
    key_facts_found: list[str] | None = None
    key_facts_missing: list[str] | None = None
    key_facts_recall: float = 0.0
    hallucinations: list[str] | None = None
    hallucination_count: int = 0
    notes: str = ""
    judge_error: str | None = None


def judge_generation(
    question: str,
    expected_answer: str,
    key_facts: list[str],
    generated_answer: str,
    judge_client: AnthropicClient,
) -> GenerationMetrics:
    """Score a generated answer using LLM-as-judge.

    Args:
        question: The evaluation question.
        expected_answer: The expected answer text.
        key_facts: List of key facts that should appear.
        generated_answer: The RAG-generated answer to evaluate.
        judge_client: AnthropicClient configured with the judge model.

    Returns:
        GenerationMetrics with scores from the judge.
    """
    metrics = GenerationMetrics(question=question)

    user_message = _JUDGE_USER_TEMPLATE.format(
        question=question,
        expected_answer=expected_answer,
        key_facts=json.dumps(key_facts),
        generated_answer=generated_answer,
    )

    try:
        response_text = judge_client.create_message(
            system=_JUDGE_SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=1024,
        )

        # Parse JSON response, stripping markdown fences if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        result = json.loads(cleaned)

        metrics.factual_accuracy = float(result.get("factual_accuracy", 0.0))
        metrics.completeness = float(result.get("completeness", 0.0))
        metrics.key_facts_found = result.get("key_facts_found", [])
        metrics.key_facts_missing = result.get("key_facts_missing", [])
        metrics.hallucinations = result.get("hallucinations", [])
        metrics.hallucination_count = len(metrics.hallucinations)
        metrics.notes = result.get("notes", "")

        if key_facts:
            metrics.key_facts_recall = len(metrics.key_facts_found or []) / len(
                key_facts
            )

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        metrics.judge_error = f"failed to parse judge response: {e}"

    return metrics


@dataclass
class AggregateGenerationMetrics:
    """Aggregated generation metrics across all questions."""

    avg_factual_accuracy: float = 0.0
    avg_completeness: float = 0.0
    avg_key_facts_recall: float = 0.0
    hallucination_rate: float = 0.0
    total_questions: int = 0
    judge_errors: int = 0
    per_question: list[GenerationMetrics] | None = None


def aggregate_generation_metrics(
    per_question: list[GenerationMetrics],
) -> AggregateGenerationMetrics:
    """Aggregate generation metrics across multiple questions."""
    if not per_question:
        return AggregateGenerationMetrics()

    n = len(per_question)
    valid = [m for m in per_question if m.judge_error is None]
    n_valid = len(valid) if valid else 1  # avoid division by zero

    total_hallucinations = sum(m.hallucination_count for m in valid)
    total_answers = sum(1 for m in valid if m.question)

    return AggregateGenerationMetrics(
        avg_factual_accuracy=sum(m.factual_accuracy for m in valid) / n_valid,
        avg_completeness=sum(m.completeness for m in valid) / n_valid,
        avg_key_facts_recall=sum(m.key_facts_recall for m in valid) / n_valid,
        hallucination_rate=(
            total_hallucinations / total_answers if total_answers else 0.0
        ),
        total_questions=n,
        judge_errors=sum(1 for m in per_question if m.judge_error is not None),
        per_question=per_question,
    )
