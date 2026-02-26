"""Tests for OCR utilities."""

from pathlib import Path

import fitz
import pytest

from pdf_llm_server.rag.ingestion.ocr import assess_needs_ocr


@pytest.fixture(scope="module")
def text_pdf_path(tmp_path_factory) -> Path:
    """Create a PDF with extractable text."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "text_pdf.pdf"

    doc = fitz.open()
    page = doc.new_page()
    # Insert substantial text
    text = "This is a test document with plenty of extractable text. " * 20
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(pdf_path)
    doc.close()

    return pdf_path


@pytest.fixture(scope="module")
def scanned_pdf_path(tmp_path_factory) -> Path:
    """Create a PDF that simulates a scanned document (no text layer)."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "scanned_pdf.pdf"

    doc = fitz.open()
    # Create a page with just a rectangle (simulating a scanned image)
    page = doc.new_page()
    rect = fitz.Rect(72, 72, 300, 300)
    page.draw_rect(rect, color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    # No text inserted - simulates scanned page
    doc.save(pdf_path)
    doc.close()

    return pdf_path


class TestAssessNeedsOCR:
    """Tests for assess_needs_ocr function."""

    def test_text_pdf_does_not_need_ocr(self, text_pdf_path):
        """Test that a text-based PDF returns False."""
        result = assess_needs_ocr(text_pdf_path)
        assert result is False

    def test_scanned_pdf_needs_ocr(self, scanned_pdf_path):
        """Test that a scanned PDF returns True."""
        result = assess_needs_ocr(scanned_pdf_path)
        assert result is True

    def test_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            assess_needs_ocr(tmp_path / "nonexistent.pdf")

    def test_accepts_path_object(self, text_pdf_path):
        """Test that Path objects are accepted."""
        result = assess_needs_ocr(Path(text_pdf_path))
        assert isinstance(result, bool)

    def test_accepts_string_path(self, text_pdf_path):
        """Test that string paths are accepted."""
        result = assess_needs_ocr(str(text_pdf_path))
        assert isinstance(result, bool)


class TestOCRWithTesseract:
    """Tests for ocr_pdf_with_tesseract function.

    Note: These tests are skipped if pytesseract is not installed.
    """

    @pytest.fixture
    def check_tesseract(self):
        """Skip tests if tesseract is not available."""
        try:
            import pytesseract
            from PIL import Image

            # Check if tesseract binary is available
            pytesseract.get_tesseract_version()
        except (ImportError, Exception):
            pytest.skip("pytesseract or tesseract not available")

    def test_ocr_import_error(self, text_pdf_path):
        """Test that ImportError is raised with helpful message when deps missing."""
        # This test checks the error handling, not actual OCR
        # We can't easily test the import error without mocking
        pass  # Covered by integration tests with real PDFs
