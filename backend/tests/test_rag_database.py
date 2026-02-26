import os
from pathlib import Path

import pytest

from pdf_llm_server.rag import PgVectorStore, ChunkRecord


# Path to migrations directory (relative to this test file)
MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


@pytest.fixture(scope="module")
def db():
    """Create a database connection for testing."""
    connection_string = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/pdf_llm_rag",
    )
    store = PgVectorStore(connection_string)
    store.connect()
    store.run_migrations(MIGRATIONS_DIR)
    yield store
    store.disconnect()


@pytest.fixture(autouse=True)
def truncate_tables(db):
    """Truncate tables before each test for isolation."""
    db.truncate_tables()
    yield


class TestDocumentOperations:
    def test_insert_document(self, db):
        doc = db.insert_document(
            file_hash="abc123hash",
            file_path="/path/to/test.pdf",
            metadata={"title": "Test Document"},
        )

        assert doc.id is not None
        assert doc.file_hash == "abc123hash"
        assert doc.file_path == "/path/to/test.pdf"
        assert doc.metadata == {"title": "Test Document"}
        assert doc.created_at is not None

    def test_get_documents(self, db):
        doc = db.insert_document(
            file_hash="hash_for_get_test",
            file_path="/path/to/get_test.pdf",
            metadata={},
        )

        documents = db.get_documents()
        assert len(documents) == 1
        assert documents[0].id == doc.id

    def test_get_document_by_hash(self, db):
        doc = db.insert_document(
            file_hash="unique_hash_123",
            file_path="/path/to/unique.pdf",
            metadata={},
        )

        found = db.get_document_by_hash("unique_hash_123")
        assert found is not None
        assert found.id == doc.id

        not_found = db.get_document_by_hash("nonexistent_hash")
        assert not_found is None

    def test_delete_document(self, db):
        doc = db.insert_document(
            file_hash="hash_to_delete",
            file_path="/path/to/delete.pdf",
            metadata={},
        )

        deleted = db.delete_document(doc.id)
        assert deleted is True

        found = db.get_document_by_hash("hash_to_delete")
        assert found is None

    def test_duplicate_hash_fails(self, db):
        db.insert_document(
            file_hash="duplicate_test_hash",
            file_path="/path/to/first.pdf",
            metadata={},
        )

        with pytest.raises(Exception):
            db.insert_document(
                file_hash="duplicate_test_hash",
                file_path="/path/to/second.pdf",
                metadata={},
            )


class TestChunkOperations:
    def test_insert_chunks(self, db):
        doc = db.insert_document(
            file_hash="hash_for_chunks",
            file_path="/path/to/chunked.pdf",
            metadata={},
        )

        chunks = [
            ChunkRecord(
                document_id=doc.id,
                content="This is the first chunk of text.",
                chunk_type="paragraph",
                page_number=1,
                position=0,
            ),
            ChunkRecord(
                document_id=doc.id,
                content="This is the second chunk of text.",
                chunk_type="paragraph",
                page_number=1,
                position=1,
            ),
        ]

        inserted = db.insert_chunks(chunks)
        assert len(inserted) == 2
        assert all(c.id is not None for c in inserted)
        assert inserted[0].content == "This is the first chunk of text."

    def test_insert_chunks_with_embedding(self, db):
        doc = db.insert_document(
            file_hash="hash_for_embedded_chunks",
            file_path="/path/to/embedded.pdf",
            metadata={},
        )

        mock_embedding = [0.1] * 1536

        chunks = [
            ChunkRecord(
                document_id=doc.id,
                content="Chunk with embedding.",
                chunk_type="paragraph",
                page_number=1,
                position=0,
                embedding=mock_embedding,
            ),
        ]

        inserted = db.insert_chunks(chunks)
        assert len(inserted) == 1
        assert inserted[0].embedding is not None


class TestSimilaritySearch:
    def test_similarity_search(self, db):
        doc = db.insert_document(
            file_hash="hash_for_search",
            file_path="/path/to/searchable.pdf",
            metadata={"source": "test"},
        )

        embedding1 = [0.1] * 1536
        embedding2 = [0.9] * 1536

        chunks = [
            ChunkRecord(
                document_id=doc.id,
                content="First searchable chunk.",
                chunk_type="paragraph",
                page_number=1,
                position=0,
                embedding=embedding1,
            ),
            ChunkRecord(
                document_id=doc.id,
                content="Second searchable chunk.",
                chunk_type="paragraph",
                page_number=2,
                position=1,
                embedding=embedding2,
            ),
        ]
        db.insert_chunks(chunks)

        query_embedding = [0.1] * 1536
        results = db.similarity_search(query_embedding, top_k=2)

        assert len(results) == 2
        assert results[0].score is not None
        assert results[0].chunk is not None
        assert results[0].document is not None

    def test_similarity_search_empty(self, db):
        # Tables are truncated before each test, so this should return empty results
        random_embedding = [0.5] * 1536
        results = db.similarity_search(random_embedding, top_k=5)
        assert results == []


class TestBm25Search:
    def test__bm25_search_returns_matching_chunks(self, db):
        doc = db.insert_document(
            file_hash="hash_for_bm25",
            file_path="/path/to/bm25.pdf",
            metadata={},
        )
        chunks = [
            ChunkRecord(
                document_id=doc.id,
                content="The plaintiff filed a class action lawsuit regarding securities fraud.",
                chunk_type="paragraph",
                page_number=1,
                position=0,
                embedding=[0.1] * 1536,
            ),
            ChunkRecord(
                document_id=doc.id,
                content="The company reported quarterly earnings for the fiscal year.",
                chunk_type="paragraph",
                page_number=2,
                position=1,
                embedding=[0.2] * 1536,
            ),
        ]
        db.insert_chunks(chunks)

        results = db._bm25_search("class action securities fraud", top_k=5)

        assert len(results) >= 1
        assert results[0].score > 0
        assert results[0].document is not None

    def test__bm25_search_empty_results(self, db):
        results = db._bm25_search("xyznonexistentterm", top_k=5)
        assert results == []

    def test__bm25_search_respects_top_k(self, db):
        doc = db.insert_document(
            file_hash="hash_bm25_topk",
            file_path="/path/to/topk.pdf",
            metadata={},
        )
        chunks = [
            ChunkRecord(
                document_id=doc.id,
                content=f"Legal document section {i} discusses legal matters.",
                chunk_type="paragraph",
                page_number=i,
                position=i,
            )
            for i in range(5)
        ]
        db.insert_chunks(chunks)

        results = db._bm25_search("legal document", top_k=2)
        assert len(results) <= 2


class TestHybridSearch:
    def test_hybrid_search_combines_results(self, db):
        doc = db.insert_document(
            file_hash="hash_hybrid",
            file_path="/path/to/hybrid.pdf",
            metadata={},
        )
        chunks = [
            ChunkRecord(
                document_id=doc.id,
                content="Securities fraud class action complaint filed in federal court.",
                chunk_type="paragraph",
                page_number=1,
                position=0,
                embedding=[0.1] * 1536,
            ),
            ChunkRecord(
                document_id=doc.id,
                content="Company financial report for quarterly earnings.",
                chunk_type="paragraph",
                page_number=2,
                position=1,
                embedding=[0.9] * 1536,
            ),
        ]
        db.insert_chunks(chunks)

        results = db.hybrid_search(
            query_embedding=[0.1] * 1536,
            query="securities fraud class action",
            top_k=5,
        )

        assert len(results) >= 1
        assert all(r.score > 0 for r in results)
        # Results should be deduplicated (no duplicate chunk IDs)
        chunk_ids = [str(r.chunk.id) for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_hybrid_search_empty_database(self, db):
        results = db.hybrid_search(
            query_embedding=[0.5] * 1536,
            query="anything",
            top_k=5,
        )
        assert results == []
