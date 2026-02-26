"""Tests for text chunking utilities."""

import pytest

from technical_rag.rag.ingestion.chunking import (
    ChunkData,
    chunk_parsed_document,
    detect_content_type,
    fixed_size_chunking,
    semantic_chunking_by_paragraphs,
)
from technical_rag.rag.ingestion.parser_models import (
    ParsedDocument,
    ParsedPage,
    TableData,
    TextBlock,
)


class TestFixedSizeChunking:
    """Tests for fixed_size_chunking function."""

    def test_empty_text(self):
        """Test with empty string."""
        result = fixed_size_chunking("")
        assert result == []

    def test_short_text(self):
        """Test text shorter than chunk size."""
        text = "Short text."
        result = fixed_size_chunking(text, chunk_size=1000)
        assert result == [text]

    def test_splits_at_word_boundary(self):
        """Test that chunks split at word boundaries."""
        text = "word " * 100  # 500 chars
        result = fixed_size_chunking(text, chunk_size=50, overlap=10)
        for chunk in result:
            # Should not end mid-word
            assert not chunk.endswith("wor")

    def test_overlap_works(self):
        """Test that overlap creates overlapping content."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        result = fixed_size_chunking(text, chunk_size=100, overlap=20)
        assert len(result) > 1
        # Check some overlap exists between consecutive chunks
        for i in range(len(result) - 1):
            # Last part of chunk i should appear in chunk i+1
            end_of_first = result[i][-15:]
            assert any(word in result[i + 1] for word in end_of_first.split())

    def test_handles_no_spaces(self):
        """Test text with no spaces."""
        text = "a" * 200
        result = fixed_size_chunking(text, chunk_size=50, overlap=10)
        assert len(result) > 1

    def test_zero_chunk_size(self):
        """Test with zero chunk size returns empty."""
        result = fixed_size_chunking("some text", chunk_size=0)
        assert result == []

    def test_negative_overlap_raises(self):
        """Test that negative overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            fixed_size_chunking("some text", chunk_size=100, overlap=-1)

    def test_overlap_exceeds_chunk_size_raises(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            fixed_size_chunking("some text", chunk_size=100, overlap=100)


class TestSemanticChunking:
    """Tests for semantic_chunking_by_paragraphs function."""

    def test_empty_text(self):
        """Test with empty string."""
        result = semantic_chunking_by_paragraphs("")
        assert result == []

    def test_single_paragraph(self):
        """Test with single paragraph."""
        text = "This is a single paragraph."
        result = semantic_chunking_by_paragraphs(text)
        assert result == [text]

    def test_preserves_paragraphs(self):
        """Test that paragraph boundaries are preserved."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = semantic_chunking_by_paragraphs(text, max_chunk_size=1000)
        # Should keep all in one chunk since total is small
        assert len(result) == 1
        assert "First paragraph" in result[0]
        assert "Second paragraph" in result[0]

    def test_splits_large_paragraphs(self):
        """Test that oversized paragraphs are split."""
        large_para = "word " * 500  # ~2500 chars
        text = f"Small intro.\n\n{large_para}\n\nSmall outro."
        result = semantic_chunking_by_paragraphs(text, max_chunk_size=500)
        # Large paragraph should be split
        assert len(result) > 2

    def test_merges_small_paragraphs(self):
        """Test that small paragraphs are merged."""
        text = "A.\n\nB.\n\nC.\n\nD."
        result = semantic_chunking_by_paragraphs(text, max_chunk_size=1000)
        # All should fit in one chunk
        assert len(result) == 1

    def test_splits_when_exceeds_max(self):
        """Test splitting when accumulated size exceeds max."""
        para = "x" * 100
        text = f"{para}\n\n{para}\n\n{para}\n\n{para}"
        result = semantic_chunking_by_paragraphs(text, max_chunk_size=250)
        # Should create multiple chunks
        assert len(result) >= 2


class TestDetectContentType:
    """Tests for detect_content_type function."""

    def test_empty_text(self):
        """Test with empty string returns paragraph."""
        assert detect_content_type("") == "paragraph"

    def test_detects_bullet_list(self):
        """Test bullet point detection."""
        text = "• Item one\n• Item two\n• Item three"
        assert detect_content_type(text) == "list"

    def test_detects_numbered_list(self):
        """Test numbered list detection."""
        text = "1. First\n2. Second\n3. Third"
        assert detect_content_type(text) == "list"

    def test_detects_dash_list(self):
        """Test dash list detection."""
        text = "- Item A\n- Item B\n- Item C"
        assert detect_content_type(text) == "list"

    def test_detects_table(self):
        """Test table-like content detection."""
        text = "Col1 | Col2 | Col3\n----|----|----|"
        assert detect_content_type(text) == "table"

    def test_detects_uppercase_heading(self):
        """Test uppercase heading detection."""
        text = "CHAPTER ONE"
        assert detect_content_type(text) == "heading"

    def test_detects_short_heading(self):
        """Test short text without punctuation as heading."""
        text = "Introduction"
        assert detect_content_type(text) == "heading"

    def test_detects_paragraph(self):
        """Test regular paragraph detection."""
        text = "This is a regular paragraph with complete sentences. It has punctuation and everything."
        assert detect_content_type(text) == "paragraph"


class TestChunkData:
    """Tests for ChunkData model."""

    def test_create_chunk_data(self):
        """Test ChunkData creation."""
        chunk = ChunkData(
            content="Test content",
            chunk_type="paragraph",
            page_number=1,
            position=0,
            bbox=[10.0, 20.0, 100.0, 50.0],
        )
        assert chunk.content == "Test content"
        assert chunk.chunk_type == "paragraph"
        assert chunk.page_number == 1
        assert chunk.position == 0
        assert chunk.bbox == [10.0, 20.0, 100.0, 50.0]

    def test_chunk_data_optional_bbox(self):
        """Test ChunkData with optional bbox."""
        chunk = ChunkData(
            content="Test",
            chunk_type="heading",
            page_number=1,
            position=0,
        )
        assert chunk.bbox is None


class TestChunkParsedDocument:
    """Tests for chunk_parsed_document grouping logic."""

    @staticmethod
    def _make_doc(
        blocks: list[tuple[str, str]],
        tables: list[TableData] | None = None,
    ) -> ParsedDocument:
        """Build a single-page ParsedDocument from (block_type, text) tuples."""
        text_blocks = [
            TextBlock(
                block_index=i,
                block_type=bt,
                text=text,
                font_size=12.0,
                is_bold=bt == "heading",
                bbox=[0.0, float(i * 20), 100.0, float(i * 20 + 15)],
            )
            for i, (bt, text) in enumerate(blocks)
        ]
        return ParsedDocument(
            file_path="test.pdf",
            total_pages=1,
            pages=[
                ParsedPage(
                    page_number=1,
                    blocks=text_blocks,
                    tables=tables or [],
                )
            ],
        )

    def test_same_type_blocks_merge(self):
        """[para, para, para] → 1 chunk."""
        doc = self._make_doc([
            ("paragraph", "First sentence."),
            ("paragraph", "Second sentence."),
            ("paragraph", "Third sentence."),
        ])
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "paragraph"

    def test_heading_splits_group(self):
        """[heading, para, para] → 2 chunks."""
        doc = self._make_doc([
            ("heading", "Title"),
            ("paragraph", "Body one."),
            ("paragraph", "Body two."),
        ])
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 2
        assert chunks[0].chunk_type == "heading"
        assert chunks[1].chunk_type == "paragraph"

    def test_mixed_body_types_merge(self):
        """[para, list_item, para] → 1 chunk (core fix)."""
        doc = self._make_doc([
            ("paragraph", "Intro text."),
            ("list_item", "1. First point."),
            ("paragraph", "Continuation text."),
        ])
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 1
        assert "Intro text." in chunks[0].content
        assert "1. First point." in chunks[0].content
        assert "Continuation text." in chunks[0].content

    def test_heading_between_paragraphs_splits(self):
        """[para, heading, para] → 3 chunks."""
        doc = self._make_doc([
            ("paragraph", "Before heading."),
            ("heading", "Section Title"),
            ("paragraph", "After heading."),
        ])
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 3
        assert chunks[0].chunk_type == "paragraph"
        assert chunks[1].chunk_type == "heading"
        assert chunks[2].chunk_type == "paragraph"

    def test_list_item_then_paragraph_merge(self):
        """[list_item, para] → 1 chunk with type=list_item."""
        doc = self._make_doc([
            ("list_item", "1. Review standard."),
            ("paragraph", "The court applies de novo review."),
        ])
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "list_item"

    def test_consecutive_headings_merge(self):
        """[heading, heading] → 1 chunk."""
        doc = self._make_doc([
            ("heading", "CHAPTER ONE"),
            ("heading", "Introduction"),
        ])
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "heading"

    def test_headings_then_paragraph_split(self):
        """[heading, heading, para] → 2 chunks."""
        doc = self._make_doc([
            ("heading", "CHAPTER ONE"),
            ("heading", "Introduction"),
            ("paragraph", "Body text here."),
        ])
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 2
        assert chunks[0].chunk_type == "heading"
        assert chunks[1].chunk_type == "paragraph"

    def test_block_bboxes_populated(self):
        """[para, list_item] → 1 chunk with 2 block_bboxes."""
        doc = self._make_doc([
            ("paragraph", "First block."),
            ("list_item", "1. Second block."),
        ])
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 1
        assert chunks[0].block_bboxes is not None
        assert len(chunks[0].block_bboxes) == 2

    def test_tables_unchanged(self):
        """Page with blocks + tables → table chunk separate with type=table."""
        table = TableData(
            table_index=0,
            headers=["Col A", "Col B"],
            rows=[["val1", "val2"]],
        )
        doc = self._make_doc(
            [("paragraph", "Some text.")],
            tables=[table],
        )
        chunks = chunk_parsed_document(doc)
        assert len(chunks) == 2
        assert chunks[0].chunk_type == "paragraph"
        assert chunks[1].chunk_type == "table"
        assert "Col A" in chunks[1].content
