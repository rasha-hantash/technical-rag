"""RAG retriever for similarity search."""

import time

from ...logger import logger
from ..database import PgVectorStore
from ..llm_clients.embeddings import EmbeddingClient
from ..models import SearchResult
from .reranker import Reranker


class RAGRetriever:
    """Retriever for RAG queries with similarity search."""

    def __init__(
        self,
        db: PgVectorStore,
        embedding_client: EmbeddingClient,
        reranker: Reranker | None = None,
    ):
        """Initialize the RAG retriever.

        Args:
            db: PgVectorStore database connection.
            embedding_client: EmbeddingClient for generating query embeddings.
            reranker: Optional Reranker instance for post-retrieval re-ranking.
        """
        self.db = db
        self.embedding_client = embedding_client
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5, tags: list[str] | None = None) -> list[SearchResult]:
        """Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Number of top results to return.
            tags: Optional tag filter — only search chunks from documents with overlapping tags.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        start = time.perf_counter()

        fetch_k = top_k * 4 if self.reranker else top_k

        query_embedding = self.embedding_client.generate_embedding(query)
        results = self.db.hybrid_search(query_embedding, query, top_k=fetch_k, tags=tags)

        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "retrieval completed",
            query_length=len(query),
            top_k=top_k,
            reranked=self.reranker is not None,
            results_count=len(results),
            duration_ms=round(duration_ms, 2),
        )

        return results
