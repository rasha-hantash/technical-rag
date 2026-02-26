"""Tests for the RAG pipeline evaluation modules."""

import json
import tempfile
from pathlib import Path

import pytest

# Add paths for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval.rag_ground_truth import (
    EvalDataset,
    EvalQuestion,
    ExpectedAnswer,
    GoldPassage,
    QuestionCategory,
    load_eval_dataset,
    save_eval_dataset,
)
from eval.pipeline_config import (
    PipelineVariant,
    get_default_matrix,
    get_quick_matrix,
)
from eval.retrieval_eval import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
    passage_in_chunk,
)


# --- passage_in_chunk tests ---


class TestPassageInChunk:
    def test_exact_substring(self):
        assert passage_in_chunk("settlement amount", "The settlement amount is $50M.")

    def test_exact_match_full(self):
        assert passage_in_chunk("hello world", "hello world")

    def test_no_match(self):
        assert not passage_in_chunk(
            "completely unrelated text",
            "The settlement was approved in 2023.",
        )

    def test_case_insensitive(self):
        assert passage_in_chunk("Settlement Amount", "the settlement amount is $50M")

    def test_whitespace_normalization(self):
        assert passage_in_chunk(
            "settlement   amount\nis  $50M",
            "The settlement amount is $50M and was approved.",
        )

    def test_fuzzy_match_minor_difference(self):
        passage = "The total settlement fund is approximately fifty million dollars"
        chunk = "The total settlement fund is approximately fifty million dollar"
        assert passage_in_chunk(passage, chunk)

    def test_fuzzy_match_below_threshold(self):
        passage = "The settlement amount is $50 million"
        chunk = "The defendant agreed to pay damages of unknown quantity"
        assert not passage_in_chunk(passage, chunk)

    def test_empty_passage(self):
        assert not passage_in_chunk("", "some chunk text")

    def test_empty_chunk(self):
        assert not passage_in_chunk("some passage", "")

    def test_passage_longer_than_chunk(self):
        passage = "This is a very long passage that exceeds the chunk"
        chunk = "short"
        assert not passage_in_chunk(passage, chunk)

    def test_passage_embedded_in_larger_chunk(self):
        passage = "class members who purchased products between January 2020 and December 2022"
        chunk = (
            "The court has certified a class consisting of all "
            "class members who purchased products between January 2020 and December 2022. "
            "Excluded from the class are officers and directors of the defendant."
        )
        assert passage_in_chunk(passage, chunk)


# --- Retrieval metrics tests ---


class TestRetrievalMetrics:
    def _make_gold(self, text: str) -> GoldPassage:
        return GoldPassage(text=text, source_pdf="test.pdf")

    def test_perfect_recall(self):
        golds = [self._make_gold("settlement amount is $50M")]
        retrieved = ["The settlement amount is $50M and was approved."]
        metrics = compute_retrieval_metrics(golds, retrieved)
        assert metrics.recall_at_k == 1.0
        assert metrics.matched_count == 1

    def test_zero_recall(self):
        golds = [self._make_gold("settlement amount is $50M")]
        retrieved = ["Completely unrelated text about weather."]
        metrics = compute_retrieval_metrics(golds, retrieved)
        assert metrics.recall_at_k == 0.0
        assert metrics.matched_count == 0

    def test_partial_recall(self):
        golds = [
            self._make_gold("settlement amount is $50M"),
            self._make_gold("opt-out deadline is March 15"),
        ]
        retrieved = ["The settlement amount is $50M.", "Something else entirely."]
        metrics = compute_retrieval_metrics(golds, retrieved)
        assert metrics.recall_at_k == 0.5
        assert metrics.matched_count == 1

    def test_mrr_first_result(self):
        golds = [self._make_gold("settlement amount")]
        retrieved = ["The settlement amount is $50M.", "Other text."]
        metrics = compute_retrieval_metrics(golds, retrieved)
        assert metrics.reciprocal_rank == 1.0

    def test_mrr_second_result(self):
        golds = [self._make_gold("settlement amount")]
        retrieved = ["Unrelated text.", "The settlement amount is $50M."]
        metrics = compute_retrieval_metrics(golds, retrieved)
        assert metrics.reciprocal_rank == 0.5

    def test_mrr_no_match(self):
        golds = [self._make_gold("settlement amount")]
        retrieved = ["No match here.", "Or here."]
        metrics = compute_retrieval_metrics(golds, retrieved)
        assert metrics.reciprocal_rank == 0.0

    def test_precision_at_k(self):
        golds = [self._make_gold("settlement amount")]
        retrieved = [
            "The settlement amount is $50M.",
            "Unrelated text.",
            "More unrelated.",
        ]
        metrics = compute_retrieval_metrics(golds, retrieved)
        assert abs(metrics.precision_at_k - 1 / 3) < 0.01

    def test_empty_gold_passages(self):
        metrics = compute_retrieval_metrics([], ["some text"])
        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.reciprocal_rank == 0.0

    def test_empty_retrieved(self):
        golds = [self._make_gold("settlement amount")]
        metrics = compute_retrieval_metrics(golds, [])
        assert metrics.recall_at_k == 0.0
        assert metrics.precision_at_k == 0.0
        assert metrics.reciprocal_rank == 0.0


class TestAggregateRetrievalMetrics:
    def test_aggregation(self):
        m1 = RetrievalMetrics(recall_at_k=1.0, precision_at_k=0.5, reciprocal_rank=1.0)
        m2 = RetrievalMetrics(recall_at_k=0.5, precision_at_k=0.25, reciprocal_rank=0.5)
        agg = aggregate_retrieval_metrics([m1, m2])
        assert agg.avg_recall_at_k == 0.75
        assert abs(agg.avg_precision_at_k - 0.375) < 0.01
        assert agg.avg_mrr == 0.75
        assert agg.total_questions == 2

    def test_empty_aggregation(self):
        agg = aggregate_retrieval_metrics([])
        assert agg.avg_recall_at_k == 0.0
        assert agg.total_questions == 0


# --- Ground truth schema tests ---


class TestEvalDataset:
    def test_create_minimal_dataset(self):
        ds = EvalDataset(
            name="test",
            pdf_corpus=["test.pdf"],
            questions=[
                EvalQuestion(
                    question="What is the settlement?",
                    expected_answer=ExpectedAnswer(answer_text="$50M"),
                    category=QuestionCategory.settlement,
                )
            ],
        )
        assert len(ds.questions) == 1
        assert ds.questions[0].category == QuestionCategory.settlement

    def test_roundtrip_save_load(self, tmp_path):
        ds = EvalDataset(
            name="roundtrip-test",
            pdf_corpus=["/path/to/test.pdf"],
            questions=[
                EvalQuestion(
                    question="Who is the defendant?",
                    gold_passages=[
                        GoldPassage(
                            text="Defendant is Acme Corp",
                            source_pdf="test.pdf",
                            page_number=1,
                        )
                    ],
                    expected_answer=ExpectedAnswer(
                        answer_text="Acme Corp",
                        key_facts=["Acme Corp"],
                    ),
                    category=QuestionCategory.parties,
                )
            ],
        )

        path = tmp_path / "test_dataset.json"
        save_eval_dataset(ds, path)

        loaded = load_eval_dataset(path)
        assert loaded.name == "roundtrip-test"
        assert len(loaded.questions) == 1
        assert loaded.questions[0].question == "Who is the defendant?"
        assert loaded.questions[0].gold_passages[0].page_number == 1
        assert loaded.questions[0].expected_answer.key_facts == ["Acme Corp"]
        assert loaded.questions[0].category == QuestionCategory.parties

    def test_validation_rejects_invalid_category(self):
        with pytest.raises(ValueError):
            EvalQuestion(
                question="test",
                expected_answer=ExpectedAnswer(answer_text="test"),
                category="invalid_category",
            )

    def test_load_classaction_v1(self):
        """Validate the seed dataset loads successfully."""
        dataset_path = (
            Path(__file__).parent.parent
            / "scripts"
            / "eval"
            / "datasets"
            / "classaction-v1.json"
        )
        if not dataset_path.exists():
            pytest.skip("Seed dataset not found")

        ds = load_eval_dataset(dataset_path)
        assert ds.name == "classaction-v1"
        assert len(ds.questions) >= 15


# --- Pipeline config tests ---


class TestPipelineConfig:
    def test_default_matrix_size(self):
        variants = get_default_matrix()
        assert len(variants) == 12

    def test_quick_matrix_size(self):
        variants = get_quick_matrix()
        assert len(variants) == 2

    def test_variant_names_unique(self):
        variants = get_default_matrix()
        names = [v.name for v in variants]
        assert len(names) == len(set(names))

    def test_quick_variants_are_subset(self):
        quick = {v.name for v in get_quick_matrix()}
        full = {v.name for v in get_default_matrix()}
        assert quick.issubset(full)

    def test_variant_label(self):
        v = PipelineVariant(
            name="test", parser="pymupdf", chunking="semantic", reranker="none"
        )
        assert v.label == "pymupdf-semantic-none"
