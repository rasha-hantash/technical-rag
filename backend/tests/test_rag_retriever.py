"""Tests for the RAG retriever and generator."""

from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from pdf_llm_server.rag import (
    RAGRetriever,
    RAGGenerator,
    RAGResponse,
    SourceReference,
    SearchResult,
    ChunkRecord,
    IngestedDocument,
    AnthropicClient,
)


@pytest.fixture
def mock_db():
    """Create a mock database."""
    db = MagicMock()
    return db


@pytest.fixture
def mock_embedding_client():
    """Create a mock embedding client."""
    client = MagicMock()
    client.generate_embedding.return_value = [0.1] * 1536
    return client


@pytest.fixture
def mock_anthropic_client():
    """Create a mock AnthropicClient."""
    client = MagicMock(spec=AnthropicClient)
    client.create_message.return_value = "This is a test answer based on the context."
    return client


@pytest.fixture
def sample_search_results():
    """Create sample search results."""
    doc1 = IngestedDocument(
        id=uuid4(),
        file_hash="abc123",
        file_path="/docs/contract.pdf",
        metadata={"type": "legal"},
        created_at=datetime.now(),
    )
    doc2 = IngestedDocument(
        id=uuid4(),
        file_hash="def456",
        file_path="/docs/report.pdf",
        metadata={"type": "report"},
        created_at=datetime.now(),
    )

    chunk1 = ChunkRecord(
        id=uuid4(),
        document_id=doc1.id,
        content="The contract states that the party of the first part shall...",
        chunk_type="paragraph",
        page_number=5,
        position=0,
        embedding=[0.1] * 1536,
    )
    chunk2 = ChunkRecord(
        id=uuid4(),
        document_id=doc2.id,
        content="According to the quarterly report, revenue increased by 15%...",
        chunk_type="paragraph",
        page_number=12,
        position=0,
        embedding=[0.2] * 1536,
    )

    return [
        SearchResult(chunk=chunk1, score=0.92, document=doc1),
        SearchResult(chunk=chunk2, score=0.85, document=doc2),
    ]


class TestRAGRetrieverInit:
    """Tests for RAGRetriever initialization."""

    def test_init(self, mock_db, mock_embedding_client):
        """Test initialization with required args."""
        retriever = RAGRetriever(
            db=mock_db,
            embedding_client=mock_embedding_client,
        )
        assert retriever.db == mock_db
        assert retriever.embedding_client == mock_embedding_client
        assert retriever.reranker is None

    def test_init_with_reranker(self, mock_db, mock_embedding_client):
        """Test initialization with a reranker."""
        mock_reranker = MagicMock()
        retriever = RAGRetriever(
            db=mock_db,
            embedding_client=mock_embedding_client,
            reranker=mock_reranker,
        )
        assert retriever.reranker == mock_reranker


class TestRetrieve:
    """Tests for the retrieve method."""

    def test_retrieve_returns_results(
        self, mock_db, mock_embedding_client, sample_search_results
    ):
        """Test that retrieve returns search results."""
        mock_db.hybrid_search.return_value = sample_search_results

        retriever = RAGRetriever(
            db=mock_db,
            embedding_client=mock_embedding_client,
        )

        results = retriever.retrieve("What does the contract say?", top_k=5)

        assert len(results) == 2
        assert results[0].score == 0.92
        mock_embedding_client.generate_embedding.assert_called_once_with(
            "What does the contract say?"
        )
        mock_db.hybrid_search.assert_called_once()

    def test_retrieve_respects_top_k(
        self, mock_db, mock_embedding_client, sample_search_results
    ):
        """Test that retrieve passes top_k to hybrid search."""
        mock_db.hybrid_search.return_value = sample_search_results[:1]

        retriever = RAGRetriever(
            db=mock_db,
            embedding_client=mock_embedding_client,
        )

        results = retriever.retrieve("query", top_k=1)

        mock_db.hybrid_search.assert_called_once()
        call_kwargs = mock_db.hybrid_search.call_args[1]
        assert call_kwargs["top_k"] == 1

    def test_retrieve_empty_results(self, mock_db, mock_embedding_client):
        """Test retrieve with no matching results."""
        mock_db.hybrid_search.return_value = []

        retriever = RAGRetriever(
            db=mock_db,
            embedding_client=mock_embedding_client,
        )

        results = retriever.retrieve("nonexistent topic")

        assert results == []


class TestRAGGeneratorInit:
    """Tests for RAGGenerator initialization."""

    def test_init(self, mock_anthropic_client):
        """Test initialization with AnthropicClient."""
        generator = RAGGenerator(anthropic_client=mock_anthropic_client)
        assert generator.anthropic_client == mock_anthropic_client
        assert generator.system_prompt == RAGGenerator.DEFAULT_SYSTEM_PROMPT

    def test_init_custom_system_prompt(self, mock_anthropic_client):
        """Test initialization with custom system prompt."""
        custom_prompt = "You are a legal assistant."
        generator = RAGGenerator(
            anthropic_client=mock_anthropic_client,
            system_prompt=custom_prompt,
        )
        assert generator.system_prompt == custom_prompt


class TestGenerate:
    """Tests for the generate method."""

    def test_generate_returns_rag_response(
        self, mock_anthropic_client, sample_search_results
    ):
        """Test that generate returns a RAGResponse."""
        generator = RAGGenerator(anthropic_client=mock_anthropic_client)

        response = generator.generate("What does the contract say?", sample_search_results)

        assert isinstance(response, RAGResponse)
        assert response.answer == "This is a test answer based on the context."
        assert response.chunks_used == 2
        assert len(response.sources) == 2

    def test_generate_builds_sources(
        self, mock_anthropic_client, sample_search_results
    ):
        """Test that generate builds source references correctly."""
        generator = RAGGenerator(anthropic_client=mock_anthropic_client)

        response = generator.generate("question", sample_search_results)

        assert len(response.sources) == 2
        assert response.sources[0].file_path == "/docs/contract.pdf"
        assert response.sources[0].page_number == 5
        assert response.sources[1].file_path == "/docs/report.pdf"
        assert response.sources[1].page_number == 12

    def test_generate_no_results(self, mock_anthropic_client):
        """Test generate with no matching results."""
        generator = RAGGenerator(anthropic_client=mock_anthropic_client)

        response = generator.generate("nonexistent topic", [])

        assert "couldn't find" in response.answer.lower()
        assert response.chunks_used == 0
        assert response.sources == []
        # Should not call Anthropic when no results
        mock_anthropic_client.create_message.assert_not_called()

    def test_generate_calls_anthropic_with_context(
        self, mock_anthropic_client, sample_search_results
    ):
        """Test that generate calls AnthropicClient with properly formatted context."""
        generator = RAGGenerator(anthropic_client=mock_anthropic_client)

        generator.generate("What does the contract say?", sample_search_results)

        mock_anthropic_client.create_message.assert_called_once()
        call_kwargs = mock_anthropic_client.create_message.call_args[1]

        assert "system" in call_kwargs
        assert "user_message" in call_kwargs

        # Check that context contains the chunk content
        user_content = call_kwargs["user_message"]
        assert "contract states" in user_content
        assert "quarterly report" in user_content


class TestSourceReference:
    """Tests for SourceReference model."""

    def test_source_reference_creation(self):
        """Test creating a SourceReference."""
        source = SourceReference(
            file_path="/path/to/doc.pdf",
            page_number=5,
            content="Full content here",
            content_preview="This is a preview...",
        )
        assert source.file_path == "/path/to/doc.pdf"
        assert source.page_number == 5
        assert source.content_preview == "This is a preview..."

    def test_source_reference_optional_page(self):
        """Test SourceReference with optional page number."""
        source = SourceReference(
            file_path="/path/to/doc.pdf",
            page_number=None,
            content="Content",
            content_preview="Preview text",
        )
        assert source.page_number is None


class TestRAGResponse:
    """Tests for RAGResponse model."""

    def test_rag_response_creation(self):
        """Test creating a RAGResponse."""
        sources = [
            SourceReference(
                file_path="/doc.pdf",
                page_number=1,
                content="content",
                content_preview="preview",
            )
        ]
        response = RAGResponse(
            answer="The answer is 42.",
            sources=sources,
            chunks_used=1,
        )
        assert response.answer == "The answer is 42."
        assert len(response.sources) == 1
        assert response.chunks_used == 1

    def test_rag_response_empty_sources(self):
        """Test RAGResponse with empty sources."""
        response = RAGResponse(
            answer="No information found.",
            sources=[],
            chunks_used=0,
        )
        assert response.sources == []
        assert response.chunks_used == 0


class TestRetrieveWithReranker:
    """Tests for retrieve with reranker integration."""

    def test_retrieve_with_reranker_overfetches(
        self, mock_db, mock_embedding_client, sample_search_results
    ):
        """Test that reranker causes over-fetching (top_k * 4)."""
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = sample_search_results[:1]
        mock_db.hybrid_search.return_value = sample_search_results

        retriever = RAGRetriever(
            db=mock_db,
            embedding_client=mock_embedding_client,
            reranker=mock_reranker,
        )

        results = retriever.retrieve("query", top_k=5)

        # hybrid_search should be called with fetch_k = 5 * 4 = 20
        call_kwargs = mock_db.hybrid_search.call_args[1]
        assert call_kwargs["top_k"] == 20

        # reranker should receive all candidates and requested top_k
        mock_reranker.rerank.assert_called_once()
        rerank_args = mock_reranker.rerank.call_args
        assert rerank_args[0][0] == "query"
        assert rerank_args[1]["top_k"] == 5

    def test_retrieve_without_reranker_no_overfetch(
        self, mock_db, mock_embedding_client, sample_search_results
    ):
        """Test that without reranker, no over-fetching occurs."""
        mock_db.hybrid_search.return_value = sample_search_results

        retriever = RAGRetriever(
            db=mock_db,
            embedding_client=mock_embedding_client,
            reranker=None,
        )

        retriever.retrieve("query", top_k=5)

        call_kwargs = mock_db.hybrid_search.call_args[1]
        assert call_kwargs["top_k"] == 5

    def test_retrieve_reranker_not_called_on_empty_results(
        self, mock_db, mock_embedding_client
    ):
        """Test that reranker is not called when search returns no results."""
        mock_reranker = MagicMock()
        mock_db.hybrid_search.return_value = []

        retriever = RAGRetriever(
            db=mock_db,
            embedding_client=mock_embedding_client,
            reranker=mock_reranker,
        )

        results = retriever.retrieve("query")

        mock_reranker.rerank.assert_not_called()
        assert results == []
