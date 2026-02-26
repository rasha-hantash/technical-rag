"""PDF parsing module using PyMuPDF for text and structure extraction."""

import os
import statistics
from pathlib import Path

import fitz  # PyMuPDF

from ...logger import logger
from .ocr import assess_needs_ocr, ocr_page
from .parser_models import ParsedDocument, ParsedPage, TableData, TextBlock
from .reducto_parser import ReductoParser

# Threshold for detecting garbage text (corrupted font encodings)
GARBAGE_CONTROL_CHAR_RATIO = 0.1  # >10% control chars = garbage


def _is_garbage_text(text: str) -> bool:
    """Detect if extracted text is binary garbage from corrupted font encodings.

    Args:
        text: The extracted text to check.

    Returns:
        True if the text appears to be garbage (high ratio of control characters).
    """
    if not text or len(text) < 20:
        return False
    # Count control characters (0x00-0x1F) excluding common whitespace
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r ")
    ratio = control_chars / len(text)
    return ratio > GARBAGE_CONTROL_CHAR_RATIO



def _extract_spans_info(block_dict: dict) -> tuple[str, float, bool]:
    """Extract text, font size, and bold status from a block's spans.

    Args:
        block_dict: A block dictionary from PyMuPDF's get_text("dict").

    Returns:
        Tuple of (text, average_font_size, is_bold).
    """
    texts = []
    font_sizes = []
    bold_count = 0
    total_spans = 0

    for line in block_dict.get("lines", []):
        for span in line.get("spans", []):
            text = span.get("text", "").strip()
            if text:
                texts.append(text)
                font_sizes.append(span.get("size", 12.0))
                total_spans += 1
                # Check for bold via font flags or font name
                flags = span.get("flags", 0)
                font_name = span.get("font", "").lower()
                if (flags & 2**4) or "bold" in font_name:
                    bold_count += 1

    if not texts:
        return "", 12.0, False

    combined_text = " ".join(texts)
    # Remove NUL characters that can occur with corrupted font encodings
    # PostgreSQL cannot store NUL (0x00) in text fields
    combined_text = combined_text.replace("\x00", "")
    avg_font_size = statistics.mean(font_sizes) if font_sizes else 12.0
    is_bold = bold_count > total_spans / 2 if total_spans > 0 else False

    return combined_text, avg_font_size, is_bold


def _classify_block(
    text: str, font_size: float, median_size: float, is_bold: bool
) -> str:
    """Classify a text block based on its properties.

    Args:
        text: The block's text content.
        font_size: Average font size of the block.
        median_size: Median font size across the document.
        is_bold: Whether the block is predominantly bold.

    Returns:
        Block type: "heading", "list_item", or "paragraph".
    """
    # Check for list items
    stripped = text.strip()
    if stripped and (
        stripped[0] in "•◦▪▸►-*"
        or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)")
    ):
        return "list_item"

    # Headings: larger font or bold with short text
    if font_size > median_size * 1.2:
        return "heading"
    if is_bold and len(text) < 100:
        return "heading"

    return "paragraph"


def parse_pdf_pymupdf(file_path: str | Path) -> ParsedDocument:
    """Parse a PDF file using PyMuPDF and extract structured content.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ParsedDocument containing all extracted pages, blocks, and tables.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    doc = fitz.open(file_path)
    try:
        logger.info("parsing pdf", file_path=str(file_path), total_pages=doc.page_count)

        # First pass: collect all font sizes to calculate median
        all_font_sizes = []
        for page in doc:
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    _, font_size, _ = _extract_spans_info(block)
                    if font_size > 0:
                        all_font_sizes.append(font_size)

        median_size = statistics.median(all_font_sizes) if all_font_sizes else 12.0

        # Second pass: extract and classify blocks
        parsed_pages = []
        ocr_pages_count = 0
        for page_num, page in enumerate(doc):
            page_dict = page.get_text("dict")
            blocks = []

            # First, extract all text to check for garbage
            page_texts = []
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:
                    text, _, _ = _extract_spans_info(block)
                    if text.strip():
                        page_texts.append(text)

            combined_page_text = " ".join(page_texts)

            # Check if this page has garbage text (corrupted font encoding)
            if _is_garbage_text(combined_page_text):
                logger.info(
                    "garbage text detected, falling back to ocr",
                    file_path=str(file_path),
                    page_number=page_num + 1,
                )
                ocr_text = ocr_page(page)
                ocr_pages_count += 1
                if ocr_text.strip():
                    # Create a single block from OCR text
                    blocks.append(
                        TextBlock(
                            block_index=0,
                            block_type="paragraph",
                            text=ocr_text.strip(),
                            font_size=median_size,
                            is_bold=False,
                            bbox=None,
                        )
                    )
            else:
                # Normal extraction
                for block_idx, block in enumerate(page_dict.get("blocks", [])):
                    if block.get("type") != 0:  # Skip non-text blocks (images, etc.)
                        continue

                    text, font_size, is_bold = _extract_spans_info(block)
                    if not text.strip():
                        continue

                    bbox = block.get("bbox")
                    if bbox is None:
                        logger.warn(
                            "missing bbox for text block",
                            file_path=str(file_path),
                            page_number=page_num + 1,
                            block_index=block_idx,
                        )

                    block_type = _classify_block(text, font_size, median_size, is_bold)

                    blocks.append(
                        TextBlock(
                            block_index=block_idx,
                            block_type=block_type,
                            text=text,
                            font_size=font_size,
                            is_bold=is_bold,
                            bbox=list(bbox) if bbox else None,
                        )
                    )

            # Extract tables using PyMuPDF's table finder
            tables = []
            try:
                page_tables = page.find_tables()
                for table_idx, table in enumerate(page_tables):
                    extracted = table.extract()
                    if extracted and len(extracted) > 0:
                        headers = [str(cell) if cell else "" for cell in extracted[0]] if extracted[0] else []
                        all_rows = extracted[1:]
                        none_count = sum(1 for row in all_rows if row is None)
                        if none_count > 0:
                            logger.debug(
                                "filtered none rows from table",
                                page_number=page_num + 1,
                                table_index=table_idx,
                                none_rows=none_count,
                                total_rows=len(all_rows),
                            )
                        rows = [
                            [str(cell) if cell else "" for cell in row]
                            for row in all_rows
                            if row is not None
                        ]
                        tables.append(
                            TableData(table_index=table_idx, headers=headers, rows=rows)
                        )
            except Exception as e:
                logger.warn(
                    "table extraction failed",
                    page_number=page_num + 1,
                    error=str(e),
                )

            parsed_pages.append(
                ParsedPage(page_number=page_num + 1, blocks=blocks, tables=tables)
            )

        logger.info(
            "pdf parsed successfully",
            file_path=str(file_path),
            total_pages=len(parsed_pages),
            total_blocks=sum(len(p.blocks) for p in parsed_pages),
            total_tables=sum(len(p.tables) for p in parsed_pages),
            ocr_pages=ocr_pages_count,
        )

        return ParsedDocument(
            file_path=str(file_path),
            total_pages=len(parsed_pages),
            pages=parsed_pages,
        )
    finally:
        doc.close()


def parse_pdf(
    file_path: str | Path,
    reducto_parser: ReductoParser | None = None,
) -> ParsedDocument:
    """Parse a PDF file using the configured parser.

    The parser is selected via the PDF_PARSER environment variable:
    - "pymupdf" (default): Uses PyMuPDF for local parsing
    - "reducto": Uses Reducto cloud API for parsing

    Args:
        file_path: Path to the PDF file.
        reducto_parser: ReductoParser instance to use when PDF_PARSER=reducto.

    For the pymupdf parser, this also assesses OCR needs and logs a warning
    if the document appears to be scanned. Reducto handles OCR internally.
    """
    parser = os.getenv("PDF_PARSER", "pymupdf").lower()

    if parser == "reducto":
        if reducto_parser is None:
            raise ValueError(
                "reducto_parser is required when PDF_PARSER=reducto. "
                "Pass a ReductoParser instance to parse_pdf()."
            )
        return reducto_parser.parse(file_path)

    if parser != "pymupdf":
        logger.warn(
            "unknown PDF_PARSER value, falling back to pymupdf",
            parser=parser,
        )

    needs_ocr = assess_needs_ocr(file_path)
    if needs_ocr:
        logger.warn(
            "document may need ocr",
            file_path=str(file_path),
            message="Text extraction may be incomplete for scanned documents",
        )

    return parse_pdf_pymupdf(file_path)
