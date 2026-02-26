"""Thin wrapper around the Cohere SDK."""

import os

try:
    import cohere
except ImportError:
    cohere = None


class CohereClient:
    """Client for Cohere's rerank API.

    Initializes the Cohere SDK once and reuses it across calls.
    """

    DEFAULT_MODEL = "rerank-v3.5"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        """Initialize the Cohere client.

        Args:
            api_key: Cohere API key. If not provided, uses COHERE_API_KEY env var.
            model: Cohere rerank model name. Defaults to rerank-v3.5.

        Raises:
            ImportError: If the cohere package is not installed.
            ValueError: If no API key is available.
        """
        if cohere is None:
            raise ImportError(
                "cohere is required for CohereClient. "
                "Install it with: pip install pdf-classaction-rag[rerank-cohere]"
            )
        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key required: provide api_key or set COHERE_API_KEY"
            )
        self._client = cohere.ClientV2(api_key=api_key)
        self.model = model or self.DEFAULT_MODEL

    def rerank(self, query: str, documents: list[str], top_n: int):
        """Rerank documents by relevance to a query.

        Args:
            query: The search query.
            documents: List of document texts to rerank.
            top_n: Number of top results to return.

        Returns:
            Cohere rerank response.
        """
        return self._client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_n,
        )
