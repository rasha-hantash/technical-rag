"""OCR utilities for scanned PDF documents."""

from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from ...logger import logger

# Threshold: pages with fewer average chars are considered scanned
SCANNED_CHARS_THRESHOLD = 50


def assess_needs_ocr(file_path: str | Path) -> bool:
    """Assess whether a PDF needs OCR processing.

    Opens the PDF and checks text extraction quality across pages.
    Returns True if the document appears to be mostly scanned/image-based.

    Args:
        file_path: Path to the PDF file.

    Returns:
        True if OCR is recommended, False if text extraction is sufficient.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc = fitz.open(file_path)
    try:
        total_chars = 0
        pages_checked = 0

        # Sample pages (first 10 or all if fewer)
        sample_size = min(10, doc.page_count)

        for i in range(sample_size):
            page = doc[i]
            text = page.get_text()
            total_chars += len(text.strip())
            pages_checked += 1

        if pages_checked == 0:
            return True

        avg_chars_per_page = total_chars / pages_checked
        needs_ocr = avg_chars_per_page < SCANNED_CHARS_THRESHOLD

        logger.info(
            "ocr assessment complete",
            file_path=str(file_path),
            avg_chars_per_page=round(avg_chars_per_page, 1),
            pages_sampled=pages_checked,
            needs_ocr=needs_ocr,
        )

        return needs_ocr
    finally:
        doc.close()


def ocr_page(page: fitz.Page, dpi: int = 200) -> str:
    """OCR a single PDF page using Tesseract.

    Args:
        page: PyMuPDF page object.
        dpi: Resolution for rendering (default 200 for balance of speed/quality).

    Returns:
        Extracted text from OCR, or empty string on timeout.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = None
    try:
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            return pytesseract.image_to_string(img, timeout=30)
        except RuntimeError:
            logger.warn("ocr timeout on page")
            return ""
    finally:
        del pix
        if img is not None:
            del img


def ocr_pdf_with_tesseract(file_path: str | Path, dpi: int = 300) -> str:
    """Perform OCR on a PDF using Tesseract.

    Args:
        file_path: Path to the PDF file.
        dpi: Resolution for rendering pages (default 300).

    Returns:
        Extracted text from all pages.

    Raises:
        ValueError: If dpi is not a positive integer.
    """
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc = fitz.open(file_path)
    try:
        all_text = []

        logger.info(
            "starting ocr processing",
            file_path=str(file_path),
            total_pages=doc.page_count,
            dpi=dpi,
        )

        for page_num, page in enumerate(doc):
            text = ocr_page(page, dpi=dpi)
            all_text.append(text)

            if (page_num + 1) % 10 == 0:
                logger.info(
                    "ocr progress",
                    pages_processed=page_num + 1,
                    total_pages=doc.page_count,
                )

        combined_text = "\n\n".join(all_text)
        logger.info(
            "ocr processing complete",
            file_path=str(file_path),
            total_chars=len(combined_text),
        )

        return combined_text
    finally:
        doc.close()
