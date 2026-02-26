"""Tests for embedding generation with mocked OpenAI API."""

import pytest
from unittest.mock import Mock, patch

from technical_rag.rag.llm_clients.embeddings import (
    EmbeddingClient,
    EmbeddingResult,
    count_tokens,
    MAX_TOKENS_PER_BATCH,
)


class TestCountTokens:
    def test_count_tokens_empty_string(self):
        assert count_tokens("") == 0

    def test_count_tokens_short_string(self):
        # tiktoken gives exact count for "hello"
        assert count_tokens("hello") == 1

    def test_count_tokens_longer_string(self):
        # 100 'a' characters - tiktoken encodes this efficiently
        text = "a" * 100
        token_count = count_tokens(text)
        assert token_count > 0
        assert token_count < 100  # Should be much less than char count

    def test_count_tokens_realistic_text(self):
        # Typical sentence - tiktoken gives precise count
        text = "The quick brown fox jumps over the lazy dog."
        token_count = count_tokens(text)
        assert token_count == 10  # Exact count from cl100k_base


class TestEmbeddingResult:
    def test_empty_result(self):
        result = EmbeddingResult()
        assert result.all_succeeded
        assert result.success_count == 0
        assert result.failure_count == 0

    def test_all_succeeded(self):
        result = EmbeddingResult(
            embeddings=[[0.1] * 1536, [0.2] * 1536],
            failed_indices=[],
            errors={},
        )
        assert result.all_succeeded
        assert result.success_count == 2
        assert result.failure_count == 0

    def test_partial_failure(self):
        result = EmbeddingResult(
            embeddings=[[0.1] * 1536, None, [0.3] * 1536],
            failed_indices=[1],
            errors={1: "Rate limit exceeded"},
        )
        assert not result.all_succeeded
        assert result.success_count == 2
        assert result.failure_count == 1

    def test_all_failed(self):
        result = EmbeddingResult(
            embeddings=[None, None],
            failed_indices=[0, 1],
            errors={0: "Error", 1: "Error"},
        )
        assert not result.all_succeeded
        assert result.success_count == 0
        assert result.failure_count == 2


class TestEmbeddingClientInit:
    def test_init_with_api_key(self):
        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai:
            _ = EmbeddingClient(api_key="test-key")
            mock_openai.assert_called_once_with(api_key="test-key")

    def test_init_from_env(self):
        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
                _ = EmbeddingClient()
                mock_openai.assert_called_once_with(api_key="env-key")

    def test_init_no_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                EmbeddingClient()


class TestGenerateEmbeddingSingle:
    def test_generate_embedding_single(self):
        """Test generating embedding for a single text."""
        mock_embedding = [0.1] * 1536

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock the response
            mock_response = Mock()
            mock_response.data = [Mock(index=0, embedding=mock_embedding)]
            mock_client.embeddings.create.return_value = mock_response

            client = EmbeddingClient(api_key="test-key")
            result = client.generate_embedding("test text")

            assert result == mock_embedding
            assert len(result) == 1536
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small",
                input=["test text"],
            )

    def test_generate_embedding_single_failure_raises(self):
        """Test that single embedding failure raises RuntimeError."""
        from openai import RateLimitError

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            with patch("technical_rag.rag.llm_clients.embeddings.time.sleep"):
                mock_client = Mock()
                mock_openai_class.return_value = mock_client

                rate_limit_error = RateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )
                mock_client.embeddings.create.side_effect = rate_limit_error

                client = EmbeddingClient(api_key="test-key")

                with pytest.raises(RuntimeError, match="Embedding generation failed"):
                    client.generate_embedding("test")


class TestGenerateEmbeddingsBatch:
    def test_generate_embeddings_batch(self):
        """Test generating embeddings for multiple texts in one batch."""
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock response with correct index ordering
            mock_response = Mock()
            mock_response.data = [
                Mock(index=0, embedding=mock_embeddings[0]),
                Mock(index=1, embedding=mock_embeddings[1]),
                Mock(index=2, embedding=mock_embeddings[2]),
            ]
            mock_client.embeddings.create.return_value = mock_response

            client = EmbeddingClient(api_key="test-key")
            texts = ["text one", "text two", "text three"]
            result = client.generate_embeddings(texts)

            assert isinstance(result, EmbeddingResult)
            assert result.all_succeeded
            assert len(result.embeddings) == 3
            assert result.embeddings == mock_embeddings
            mock_client.embeddings.create.assert_called_once()

    def test_generate_embeddings_empty_list(self):
        """Test that empty input returns empty result."""
        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI"):
            client = EmbeddingClient(api_key="test-key")
            result = client.generate_embeddings([])
            assert isinstance(result, EmbeddingResult)
            assert result.embeddings == []
            assert result.all_succeeded

    def test_generate_embeddings_preserves_order(self):
        """Test that embeddings are returned in input order even if API returns out of order."""
        mock_embeddings = [[0.1] * 1536, [0.2] * 1536]

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Mock response with reversed index order
            mock_response = Mock()
            mock_response.data = [
                Mock(index=1, embedding=mock_embeddings[1]),  # Second returned first
                Mock(index=0, embedding=mock_embeddings[0]),
            ]
            mock_client.embeddings.create.return_value = mock_response

            client = EmbeddingClient(api_key="test-key")
            result = client.generate_embeddings(["first", "second"])

            # Should be in original order
            assert result.embeddings[0] == mock_embeddings[0]
            assert result.embeddings[1] == mock_embeddings[1]


class TestGenerateEmbeddingsLargeBatchSplits:
    def test_generate_embeddings_large_batch_splits(self):
        """Test that large batches are split based on token count."""
        # Create texts that will exceed MAX_TOKENS_PER_BATCH (8191 tokens)
        # Each text needs to be large enough that 2 texts exceed the limit
        # Using repeated words to get predictable token counts
        large_text = "hello world " * 2000  # ~4000 tokens each
        texts = [large_text, large_text, large_text]  # ~12000 tokens total, needs 2 batches

        mock_embedding = [0.1] * 1536

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Each call returns embeddings for that batch
            def create_response(*args, **kwargs):
                input_texts = kwargs.get("input", [])
                mock_response = Mock()
                mock_response.data = [
                    Mock(index=i, embedding=mock_embedding)
                    for i in range(len(input_texts))
                ]
                return mock_response

            mock_client.embeddings.create.side_effect = create_response

            client = EmbeddingClient(api_key="test-key")
            result = client.generate_embeddings(texts)

            assert isinstance(result, EmbeddingResult)
            assert result.all_succeeded
            assert len(result.embeddings) == 3
            # Should have been called multiple times due to batching
            assert mock_client.embeddings.create.call_count >= 2


class TestRetryAndPartialFailure:
    def test_retry_on_rate_limit_then_succeed(self):
        """Test exponential backoff retry on 429 rate limit errors."""
        from openai import RateLimitError

        mock_embedding = [0.1] * 1536

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            with patch("technical_rag.rag.llm_clients.embeddings.time.sleep") as mock_sleep:
                mock_client = Mock()
                mock_openai_class.return_value = mock_client

                # Fail twice with rate limit, then succeed
                mock_response = Mock()
                mock_response.data = [Mock(index=0, embedding=mock_embedding)]

                rate_limit_error = RateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )

                mock_client.embeddings.create.side_effect = [
                    rate_limit_error,
                    rate_limit_error,
                    mock_response,
                ]

                client = EmbeddingClient(api_key="test-key")
                result = client.generate_embeddings(["test"])

                assert result.all_succeeded
                assert result.embeddings[0] == mock_embedding
                assert mock_client.embeddings.create.call_count == 3
                # Check exponential backoff: 1s, 2s
                assert mock_sleep.call_count == 2
                mock_sleep.assert_any_call(1)
                mock_sleep.assert_any_call(2)

    def test_retry_on_server_error_then_succeed(self):
        """Test retry on 5xx server errors."""
        from openai import APIStatusError

        mock_embedding = [0.1] * 1536

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            with patch("technical_rag.rag.llm_clients.embeddings.time.sleep") as mock_sleep:
                mock_client = Mock()
                mock_openai_class.return_value = mock_client

                mock_response = Mock()
                mock_response.data = [Mock(index=0, embedding=mock_embedding)]

                # Create a 500 error
                server_error = APIStatusError(
                    message="Internal server error",
                    response=Mock(status_code=500),
                    body={"error": {"message": "Internal server error"}},
                )
                server_error.status_code = 500

                mock_client.embeddings.create.side_effect = [
                    server_error,
                    mock_response,
                ]

                client = EmbeddingClient(api_key="test-key")
                result = client.generate_embeddings(["test"])

                assert result.all_succeeded
                assert result.embeddings[0] == mock_embedding
                assert mock_client.embeddings.create.call_count == 2

    def test_client_error_no_retry_records_failure(self):
        """Test that 4xx errors (except 429) record failure without retrying."""
        from openai import APIStatusError

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            # Create a 400 bad request error
            client_error = APIStatusError(
                message="Bad request",
                response=Mock(status_code=400),
                body={"error": {"message": "Bad request"}},
            )
            client_error.status_code = 400

            mock_client.embeddings.create.side_effect = client_error

            client = EmbeddingClient(api_key="test-key")
            result = client.generate_embeddings(["test"])

            # Should record failure, not raise
            assert not result.all_succeeded
            assert result.failed_indices == [0]
            assert 0 in result.errors
            assert result.embeddings[0] is None
            # Should only be called once (no retries)
            assert mock_client.embeddings.create.call_count == 1

    def test_max_retries_exhausted_records_failure(self):
        """Test that failure is recorded after max retries."""
        from openai import RateLimitError

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            with patch("technical_rag.rag.llm_clients.embeddings.time.sleep"):
                mock_client = Mock()
                mock_openai_class.return_value = mock_client

                rate_limit_error = RateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )

                # Always fail
                mock_client.embeddings.create.side_effect = rate_limit_error

                client = EmbeddingClient(api_key="test-key")
                result = client.generate_embeddings(["test"])

                # Should record failure, not raise
                assert not result.all_succeeded
                assert result.failed_indices == [0]
                assert 0 in result.errors
                assert result.embeddings[0] is None
                # Should try exactly MAX_RETRIES times (3)
                assert mock_client.embeddings.create.call_count == 3

    def test_partial_batch_failure(self):
        """Test that some batches can succeed while others fail."""
        from openai import RateLimitError

        mock_embedding = [0.1] * 1536
        # Create 3 texts that will be split into multiple batches
        # Each text is ~4000 tokens, so with 8191 limit we get:
        # - Batch 1: text 0 and 1 (~8000 tokens)
        # - Batch 2: text 2 (~4000 tokens)
        large_text = "hello world " * 2000  # ~4000 tokens
        texts = [large_text, large_text, large_text]

        with patch("technical_rag.rag.llm_clients.embeddings.OpenAI") as mock_openai_class:
            with patch("technical_rag.rag.llm_clients.embeddings.time.sleep"):
                mock_client = Mock()
                mock_openai_class.return_value = mock_client

                # First batch (2 texts) succeeds, second batch always fails
                mock_response_batch1 = Mock()
                mock_response_batch1.data = [
                    Mock(index=0, embedding=mock_embedding),
                    Mock(index=1, embedding=mock_embedding),
                ]

                rate_limit_error = RateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(status_code=429),
                    body={"error": {"message": "Rate limit exceeded"}},
                )

                # Batch 1: success (2 texts)
                # Batch 2: fail, fail, fail (exhausts retries)
                mock_client.embeddings.create.side_effect = [
                    mock_response_batch1,  # Batch 1 succeeds
                    rate_limit_error,  # Batch 2 retry 1
                    rate_limit_error,  # Batch 2 retry 2
                    rate_limit_error,  # Batch 2 retry 3
                ]

                client = EmbeddingClient(api_key="test-key")
                result = client.generate_embeddings(texts)

                # Partial success: texts 0 and 1 succeeded, text 2 failed
                assert not result.all_succeeded
                assert result.success_count == 2
                assert result.failure_count == 1
                assert 2 in result.failed_indices
                assert result.embeddings[0] == mock_embedding
                assert result.embeddings[1] == mock_embedding
                assert result.embeddings[2] is None
