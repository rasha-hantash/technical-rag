"""Re-ranking implementations for RAG retrieval results."""

import time
from abc import ABC, abstractmethod

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from ...logger import logger
from ..llm_clients.cohere_client import CohereClient
from ..models import SearchResult


class Reranker(ABC):
    """Abstract base class for re-ranking search results."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-rank search results by relevance to the query.

        Args:
            query: The original search query.
            results: List of SearchResult candidates to re-rank.
            top_k: Number of top results to return after re-ranking.

        Returns:
            List of SearchResult objects re-ordered by relevance, truncated to top_k.
        """


class CohereReranker(Reranker):
    """Re-ranks search results using Cohere's rerank API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the Cohere re-ranker.

        Args:
            api_key: Cohere API key. If not provided, uses COHERE_API_KEY env var.
            model: Cohere rerank model name. Defaults to rerank-v3.5.

        Raises:
            ImportError: If the cohere package is not installed.
            ValueError: If no API key is available.
        """
        self._client = CohereClient(api_key=api_key, model=model)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        start = time.perf_counter()

        documents = [r.chunk.content for r in results]

        response = self._client.rerank(query, documents, top_n=top_k)

        reranked = []
        for item in response.results:
            original = results[item.index]
            reranked.append(
                SearchResult(
                    chunk=original.chunk,
                    score=item.relevance_score,
                    document=original.document,
                )
            )

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "cohere reranking completed",
            query_length=len(query),
            candidates=len(results),
            top_k=top_k,
            results_count=len(reranked),
            model=self._client.model,
            duration_ms=round(duration_ms, 2),
        )

        return reranked


class CrossEncoderReranker(Reranker):
    """Re-ranks search results using a local cross-encoder model."""

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None):
        """Initialize the cross-encoder re-ranker.

        Args:
            model_name: HuggingFace model name. Defaults to ms-marco-MiniLM-L-6-v2.
        """
        if CrossEncoder is None:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install pdf-classaction-rag[rerank-local]"
            )
        self._model_name = model_name or self.DEFAULT_MODEL
        start = time.perf_counter()
        self._model = CrossEncoder(self._model_name)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "cross-encoder model loaded",
            model=self._model_name,
            duration_ms=round(duration_ms, 2),
        )

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        start = time.perf_counter()

        pairs = [(query, r.chunk.content) for r in results]
        scores = self._model.predict(pairs)

        scored = list(zip(scores, results))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_k]

        reranked = [
            SearchResult(
                chunk=result.chunk,
                score=float(score),
                document=result.document,
            )
            for score, result in scored
        ]

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "cross-encoder reranking completed",
            query_length=len(query),
            candidates=len(results),
            top_k=top_k,
            results_count=len(reranked),
            model=self._model_name,
            duration_ms=round(duration_ms, 2),
        )

        return reranked
