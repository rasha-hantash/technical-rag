"""Tests for the re-ranking implementations."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from pdf_llm_server.rag import (
    CohereReranker,
    CrossEncoderReranker,
    SearchResult,
    ChunkRecord,
    IngestedDocument,
)


@pytest.fixture
def sample_results():
    """Create sample search results for reranking."""
    doc = IngestedDocument(
        id=uuid4(),
        file_hash="abc123",
        file_path="/docs/test.pdf",
        metadata={},
        created_at=datetime.now(),
    )
    return [
        SearchResult(
            chunk=ChunkRecord(
                id=uuid4(),
                document_id=doc.id,
                content="The defendant violated securities law section 10b.",
                chunk_type="paragraph",
                page_number=1,
                position=0,
            ),
            score=0.8,
            document=doc,
        ),
        SearchResult(
            chunk=ChunkRecord(
                id=uuid4(),
                document_id=doc.id,
                content="The weather forecast predicts rain tomorrow.",
                chunk_type="paragraph",
                page_number=2,
                position=1,
            ),
            score=0.7,
            document=doc,
        ),
        SearchResult(
            chunk=ChunkRecord(
                id=uuid4(),
                document_id=doc.id,
                content="Securities fraud investigation revealed systematic violations.",
                chunk_type="paragraph",
                page_number=3,
                position=2,
            ),
            score=0.6,
            document=doc,
        ),
    ]


class TestCohereRerankerInit:
    def test_init_with_api_key(self):
        with patch("pdf_llm_server.rag.retrieval.reranker.CohereClient") as mock_cls:
            CohereReranker(api_key="test-key")
            mock_cls.assert_called_once_with(api_key="test-key", model=None)

    def test_init_from_env(self):
        with patch("pdf_llm_server.rag.retrieval.reranker.CohereClient") as mock_cls:
            CohereReranker()
            mock_cls.assert_called_once_with(api_key=None, model=None)

    def test_init_no_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("COHERE_API_KEY", None)
            with pytest.raises((ValueError, ImportError)):
                CohereReranker()

    def test_init_custom_model(self):
        with patch("pdf_llm_server.rag.retrieval.reranker.CohereClient") as mock_cls:
            CohereReranker(api_key="test-key", model="rerank-english-v2.0")
            mock_cls.assert_called_once_with(api_key="test-key", model="rerank-english-v2.0")


class TestCohereRerank:
    def test_rerank_reorders_results(self, sample_results):
        with patch("pdf_llm_server.rag.retrieval.reranker.CohereClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.model = "rerank-v3.5"
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.results = [
                MagicMock(index=2, relevance_score=0.95),
                MagicMock(index=0, relevance_score=0.82),
            ]
            mock_client.rerank.return_value = mock_response

            reranker = CohereReranker(api_key="test-key")
            reranked = reranker.rerank("securities fraud violations", sample_results, top_k=2)

            assert len(reranked) == 2
            assert "systematic violations" in reranked[0].chunk.content
            assert reranked[0].score == 0.95
            assert "defendant violated" in reranked[1].chunk.content
            assert reranked[1].score == 0.82

    def test_rerank_empty_results(self):
        with patch("pdf_llm_server.rag.retrieval.reranker.CohereClient"):
            reranker = CohereReranker(api_key="test-key")
            results = reranker.rerank("query", [], top_k=5)
            assert results == []

    def test_rerank_passes_correct_documents(self, sample_results):
        with patch("pdf_llm_server.rag.retrieval.reranker.CohereClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.model = "rerank-v3.5"
            mock_cls.return_value = mock_client

            mock_response = MagicMock()
            mock_response.results = [MagicMock(index=0, relevance_score=0.9)]
            mock_client.rerank.return_value = mock_response

            reranker = CohereReranker(api_key="test-key")
            reranker.rerank("query", sample_results, top_k=1)

            mock_client.rerank.assert_called_once()
            call_args = mock_client.rerank.call_args
            assert call_args[0][0] == "query"
            assert len(call_args[0][1]) == 3
            assert call_args[1]["top_n"] == 1


class TestCrossEncoderRerankerInit:
    def test_init_default_model(self):
        with patch("pdf_llm_server.rag.retrieval.reranker.CrossEncoder") as mock_ce:
            reranker = CrossEncoderReranker()
            mock_ce.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def test_init_custom_model(self):
        with patch("pdf_llm_server.rag.retrieval.reranker.CrossEncoder") as mock_ce:
            reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
            mock_ce.assert_called_once_with("cross-encoder/ms-marco-TinyBERT-L-2-v2")


class TestCrossEncoderRerank:
    def test_rerank_reorders_results(self, sample_results):
        with patch("pdf_llm_server.rag.retrieval.reranker.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_ce.return_value = mock_model
            # Return scores: index 2 highest, index 0 middle, index 1 lowest
            mock_model.predict.return_value = [0.6, 0.1, 0.9]

            reranker = CrossEncoderReranker()
            reranked = reranker.rerank("securities fraud", sample_results, top_k=2)

            assert len(reranked) == 2
            assert "systematic violations" in reranked[0].chunk.content
            assert reranked[0].score == 0.9
            assert "defendant violated" in reranked[1].chunk.content
            assert reranked[1].score == 0.6

    def test_rerank_empty_results(self):
        with patch("pdf_llm_server.rag.retrieval.reranker.CrossEncoder"):
            reranker = CrossEncoderReranker()
            results = reranker.rerank("query", [], top_k=5)
            assert results == []

    def test_rerank_passes_correct_pairs(self, sample_results):
        with patch("pdf_llm_server.rag.retrieval.reranker.CrossEncoder") as mock_ce:
            mock_model = MagicMock()
            mock_ce.return_value = mock_model
            mock_model.predict.return_value = [0.9, 0.5, 0.3]

            reranker = CrossEncoderReranker()
            reranker.rerank("my query", sample_results, top_k=3)

            call_args = mock_model.predict.call_args[0][0]
            assert len(call_args) == 3
            assert all(pair[0] == "my query" for pair in call_args)
            assert call_args[0][1] == "The defendant violated securities law section 10b."
