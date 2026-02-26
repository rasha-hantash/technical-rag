"""Shared data models for PDF parsers."""

from pydantic import BaseModel


class TextBlock(BaseModel):
    """A text block extracted from a PDF page."""

    block_index: int
    block_type: str  # "heading", "paragraph", "list_item"
    text: str
    font_size: float
    is_bold: bool
    bbox: list[float] | None = None  # [x0, y0, x1, y1], None if unavailable


class TableData(BaseModel):
    """Table data extracted from a PDF page."""

    table_index: int
    headers: list[str]
    rows: list[list[str]]


class ParsedPage(BaseModel):
    """A parsed PDF page containing blocks and tables."""

    page_number: int
    blocks: list[TextBlock]
    tables: list[TableData]


class ParsedDocument(BaseModel):
    """A fully parsed PDF document."""

    file_path: str
    total_pages: int
    pages: list[ParsedPage]
