"""Tests for PDF parser module."""

from pathlib import Path

import fitz  # PyMuPDF
import pytest

from pdf_llm_server.rag.ingestion.pdf_parser import (
    ParsedDocument,
    ParsedPage,
    TextBlock,
    _classify_block,
    _extract_spans_info,
    parse_pdf,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def sample_pdf_path(tmp_path_factory) -> Path:
    """Create a sample PDF for testing."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "sample.pdf"

    doc = fitz.open()

    # Page 1: Title and paragraphs
    page = doc.new_page()
    # Large bold title
    page.insert_text(
        (72, 72),
        "Sample Document Title",
        fontsize=24,
        fontname="helv",
    )
    # Regular paragraph
    page.insert_text(
        (72, 120),
        "This is a regular paragraph of text. It contains multiple sentences "
        "that provide context and information about the document.",
        fontsize=12,
        fontname="helv",
    )
    # Another paragraph
    page.insert_text(
        (72, 160),
        "This is another paragraph with different content.",
        fontsize=12,
        fontname="helv",
    )
    # List items
    page.insert_text((72, 200), "• First bullet point", fontsize=12, fontname="helv")
    page.insert_text((72, 220), "• Second bullet point", fontsize=12, fontname="helv")

    # Page 2: More content
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Section Header", fontsize=18, fontname="helv")
    page2.insert_text(
        (72, 120),
        "More detailed content on the second page.",
        fontsize=12,
        fontname="helv",
    )

    doc.save(pdf_path)
    doc.close()

    return pdf_path


class TestPDFParser:
    """Tests for parse_pdf function."""

    def test_parse_pdf_returns_parsed_document(self, sample_pdf_path):
        """Test that parse_pdf returns a ParsedDocument."""
        result = parse_pdf(sample_pdf_path)
        assert isinstance(result, ParsedDocument)
        assert result.total_pages == 2
        assert len(result.pages) == 2

    def test_parse_pdf_extracts_file_path(self, sample_pdf_path):
        """Test that file path is captured."""
        result = parse_pdf(sample_pdf_path)
        assert str(sample_pdf_path) in result.file_path

    def test_parse_pdf_extracts_blocks(self, sample_pdf_path):
        """Test that text blocks are extracted."""
        result = parse_pdf(sample_pdf_path)
        page1 = result.pages[0]
        assert len(page1.blocks) > 0
        for block in page1.blocks:
            assert isinstance(block, TextBlock)
            assert block.text
            assert block.font_size > 0
            assert len(block.bbox) == 4

    def test_parse_pdf_detects_headings(self, sample_pdf_path):
        """Test that larger text is classified as headings."""
        result = parse_pdf(sample_pdf_path)
        # The title with 24pt font should be detected as heading
        headings = [b for b in result.pages[0].blocks if b.block_type == "heading"]
        assert len(headings) >= 1

    def test_parse_pdf_detects_list_items(self, sample_pdf_path):
        """Test that bullet points are classified as list items."""
        result = parse_pdf(sample_pdf_path)
        list_items = [b for b in result.pages[0].blocks if b.block_type == "list_item"]
        assert len(list_items) >= 2

    def test_parse_pdf_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_pdf(tmp_path / "nonexistent.pdf")

    def test_parse_pdf_multiple_pages(self, sample_pdf_path):
        """Test parsing of multi-page documents."""
        result = parse_pdf(sample_pdf_path)
        assert result.total_pages == 2
        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2


class TestBlockClassification:
    """Tests for block classification logic."""

    def test_classify_heading_by_size(self):
        """Test that larger font triggers heading classification."""
        result = _classify_block("Large Title", font_size=18.0, median_size=12.0, is_bold=False)
        assert result == "heading"

    def test_classify_heading_by_bold_short_text(self):
        """Test that bold short text is classified as heading."""
        result = _classify_block("Bold Title", font_size=12.0, median_size=12.0, is_bold=True)
        assert result == "heading"

    def test_classify_paragraph(self):
        """Test regular text is classified as paragraph."""
        long_text = "This is a longer paragraph that should not be classified as a heading."
        result = _classify_block(long_text, font_size=12.0, median_size=12.0, is_bold=False)
        assert result == "paragraph"

    def test_classify_list_bullet(self):
        """Test bullet point classification."""
        result = _classify_block("• Bullet item", font_size=12.0, median_size=12.0, is_bold=False)
        assert result == "list_item"

    def test_classify_list_numbered(self):
        """Test numbered list classification."""
        result = _classify_block("1. First item", font_size=12.0, median_size=12.0, is_bold=False)
        assert result == "list_item"

    def test_classify_list_dash(self):
        """Test dash list classification."""
        result = _classify_block("- Dash item", font_size=12.0, median_size=12.0, is_bold=False)
        assert result == "list_item"


class TestSpansExtraction:
    """Tests for span info extraction."""

    def test_extract_empty_block(self):
        """Test extraction from empty block."""
        text, size, bold = _extract_spans_info({})
        assert text == ""
        assert size == 12.0
        assert bold is False

    def test_extract_simple_block(self):
        """Test extraction from simple block."""
        block = {
            "lines": [
                {"spans": [{"text": "Hello", "size": 14.0, "flags": 0, "font": "Arial"}]}
            ]
        }
        text, size, bold = _extract_spans_info(block)
        assert text == "Hello"
        assert size == 14.0
        assert bold is False

    def test_extract_bold_by_flags(self):
        """Test bold detection via flags."""
        block = {
            "lines": [
                {"spans": [{"text": "Bold", "size": 12.0, "flags": 16, "font": "Arial"}]}
            ]
        }
        text, size, bold = _extract_spans_info(block)
        assert bold is True

    def test_extract_bold_by_font_name(self):
        """Test bold detection via font name."""
        block = {
            "lines": [
                {"spans": [{"text": "Bold", "size": 12.0, "flags": 0, "font": "Arial-Bold"}]}
            ]
        }
        text, size, bold = _extract_spans_info(block)
        assert bold is True
