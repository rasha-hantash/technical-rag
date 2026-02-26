"""Embedding generation client for OpenAI text-embedding-3-large model."""

import os
import time
from dataclasses import dataclass, field

import tiktoken
from openai import OpenAI, RateLimitError, APIStatusError

from ...logger import logger

# Constants
MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
MAX_TOKENS_PER_BATCH = 8191  # OpenAI's limit for text-embedding-3-large
MAX_RETRIES = 3
INITIAL_RETRY_DELAY_SECONDS = 1

# Tokenizer for accurate token counting (text-embedding-3-large uses cl100k_base)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens for a text string using tiktoken.

    Uses the cl100k_base encoding which is used by text-embedding-3-large.

    Args:
        text: The text to count tokens for.

    Returns:
        Exact token count.
    """
    return len(_tokenizer.encode(text))


@dataclass
class BatchResult:
    """Result of a single batch embedding generation.

    Attributes:
        embeddings: List of embedding vectors. None for texts that failed.
        error: Error message if the batch failed, None otherwise.
    """

    embeddings: list[list[float] | None]
    error: str | None


@dataclass
class EmbeddingResult:
    """Result of embedding generation with support for partial failures.

    Attributes:
        embeddings: List of embedding vectors. None for texts that failed.
        failed_indices: Indices of texts that failed to embed.
        errors: Mapping from failed index to error message.
    """

    embeddings: list[list[float] | None] = field(default_factory=list)
    failed_indices: list[int] = field(default_factory=list)
    errors: dict[int, str] = field(default_factory=dict)

    @property
    def all_succeeded(self) -> bool:
        """Return True if all texts were successfully embedded."""
        return len(self.failed_indices) == 0

    @property
    def success_count(self) -> int:
        """Return the number of successfully embedded texts."""
        return len(self.embeddings) - len(self.failed_indices)

    @property
    def failure_count(self) -> int:
        """Return the number of failed texts."""
        return len(self.failed_indices)


class EmbeddingClient:
    """Client for generating embeddings using OpenAI's API."""

    def __init__(self, api_key: str | None = None):
        """Initialize the embedding client.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required: provide api_key or set OPENAI_API_KEY"
            )
        self._client = OpenAI(api_key=api_key)

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to generate an embedding for.

        Returns:
            Embedding vector of 3072 floats.

        Raises:
            Exception: If embedding generation fails after retries.
        """
        result = self.generate_embeddings([text])
        if result.failed_indices:
            raise RuntimeError(f"Embedding generation failed: {result.errors[0]}")
        return result.embeddings[0]

    def generate_embeddings(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for a batch of texts.

        Automatically batches requests to stay within token limits and
        implements exponential backoff retry on rate limit/server errors.
        Returns partial results on failure instead of raising.

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            EmbeddingResult with embeddings (None for failures), failed indices,
            and error messages. Order matches input texts.
        """
        if not texts:
            return EmbeddingResult()

        # Split into batches based on estimated token count
        batches = self._split_into_batches(texts)

        # Track which original indices are in each batch
        batch_indices = self._get_batch_indices(texts, batches)

        result = EmbeddingResult(embeddings=[None] * len(texts))

        for batch_idx, (batch, indices) in enumerate(zip(batches, batch_indices)):
            batch_result = self._generate_batch_with_retry(
                batch, indices, batch_idx, len(batches)
            )

            # Merge batch results into overall result
            for i, embedding in zip(indices, batch_result.embeddings):
                result.embeddings[i] = embedding

            if batch_result.error:
                for i in indices:
                    result.failed_indices.append(i)
                    result.errors[i] = batch_result.error

        return result

    def _split_into_batches(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches that fit within token limits.

        Args:
            texts: List of texts to batch.

        Returns:
            List of batches, where each batch is a list of texts.
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = count_tokens(text)

            # If single text exceeds limit, it gets its own batch
            if text_tokens >= MAX_TOKENS_PER_BATCH:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                batches.append([text])
                continue

            # Check if adding this text would exceed batch limit
            if current_tokens + text_tokens > MAX_TOKENS_PER_BATCH:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def _get_batch_indices(
        self, texts: list[str], batches: list[list[str]]
    ) -> list[list[int]]:
        """Get the original indices for each text in each batch.

        Args:
            texts: Original list of texts.
            batches: List of batches.

        Returns:
            List of index lists, one per batch.
        """
        batch_indices = []
        current_idx = 0

        for batch in batches:
            indices = list(range(current_idx, current_idx + len(batch)))
            batch_indices.append(indices)
            current_idx += len(batch)

        return batch_indices

    def _generate_batch_with_retry(
        self,
        texts: list[str],
        original_indices: list[int],
        batch_idx: int,
        total_batches: int,
    ) -> BatchResult:
        """Generate embeddings for a batch with exponential backoff retry.

        Args:
            texts: Batch of texts to embed.
            original_indices: Original indices of these texts.
            batch_idx: Index of current batch (for logging).
            total_batches: Total number of batches (for logging).

        Returns:
            BatchResult with embeddings (None for failures) and error message.
        """
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                start = time.perf_counter()
                response = self._client.embeddings.create(
                    model=MODEL,
                    input=texts,
                )
                duration_ms = (time.perf_counter() - start) * 1000

                # Extract embeddings in correct order (response includes index)
                embeddings = [None] * len(texts)
                for item in response.data:
                    embeddings[item.index] = item.embedding

                logger.info(
                    "embeddings generated",
                    batch=f"{batch_idx + 1}/{total_batches}",
                    texts_count=len(texts),
                    model=MODEL,
                    duration_ms=round(duration_ms, 2),
                )

                return BatchResult(embeddings=embeddings, error=None)

            except RateLimitError as e:
                last_error = str(e)
                delay = INITIAL_RETRY_DELAY_SECONDS * (2**attempt)  # 1s, 2s, 4s
                logger.warn(
                    "rate limit hit, retrying",
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    delay_seconds=delay,
                    error=last_error,
                )
                time.sleep(delay)

            except APIStatusError as e:
                if e.status_code >= 500:
                    last_error = str(e)
                    delay = INITIAL_RETRY_DELAY_SECONDS * (2**attempt)
                    logger.warn(
                        "server error, retrying",
                        attempt=attempt + 1,
                        max_retries=MAX_RETRIES,
                        delay_seconds=delay,
                        status_code=e.status_code,
                        error=last_error,
                    )
                    time.sleep(delay)
                else:
                    # 4xx errors (except 429) should not be retried
                    last_error = str(e)
                    logger.error(
                        "embedding generation failed",
                        batch=f"{batch_idx + 1}/{total_batches}",
                        status_code=e.status_code,
                        error=last_error,
                    )
                    return BatchResult(embeddings=[None] * len(texts), error=last_error)

            except Exception as e:
                last_error = str(e)
                logger.error(
                    "unexpected error during embedding generation",
                    batch=f"{batch_idx + 1}/{total_batches}",
                    error=last_error,
                )
                return BatchResult(embeddings=[None] * len(texts), error=last_error)

        # All retries exhausted
        logger.error(
            "embedding generation failed after retries",
            batch=f"{batch_idx + 1}/{total_batches}",
            max_retries=MAX_RETRIES,
            error=last_error,
        )
        return BatchResult(embeddings=[None] * len(texts), error=last_error)


# Convenience functions for module-level access
_default_client: EmbeddingClient | None = None


def _get_default_client() -> EmbeddingClient:
    """Get or create the default embedding client."""
    global _default_client
    if _default_client is None:
        _default_client = EmbeddingClient()
    return _default_client


def generate_embedding(text: str) -> list[float]:
    """Generate embedding for a single text using the default client.

    Args:
        text: The text to generate an embedding for.

    Returns:
        Embedding vector of 3072 floats.

    Raises:
        RuntimeError: If embedding generation fails after retries.
    """
    return _get_default_client().generate_embedding(text)


def generate_embeddings(texts: list[str]) -> EmbeddingResult:
    """Generate embeddings for multiple texts using the default client.

    Args:
        texts: List of texts to generate embeddings for.

    Returns:
        EmbeddingResult with embeddings, failed indices, and errors.
    """
    return _get_default_client().generate_embeddings(texts)
