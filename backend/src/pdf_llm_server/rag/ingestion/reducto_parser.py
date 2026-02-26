"""PDF parsing module using Reducto cloud API for text and structure extraction."""

import os
import time
from html.parser import HTMLParser
from pathlib import Path

try:
    from reducto.reducto import Reducto
except ImportError:
    Reducto = None

from ...logger import logger
from .parser_models import ParsedDocument, ParsedPage, TableData, TextBlock


class _TableHTMLParser(HTMLParser):
    """Parse HTML table content into headers and rows."""

    def __init__(self):
        super().__init__()
        self.headers: list[str] = []
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: str = ""
        self._in_th = False
        self._in_td = False

    def handle_starttag(self, tag, attrs):
        if tag == "th":
            self._in_th = True
            self._current_cell = ""
        elif tag == "td":
            self._in_td = True
            self._current_cell = ""
        elif tag == "tr":
            self._current_row = []

    def handle_endtag(self, tag):
        if tag == "th":
            self._in_th = False
            self.headers.append(self._current_cell.strip())
        elif tag == "td":
            self._in_td = False
            self._current_row.append(self._current_cell.strip())
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)

    def handle_data(self, data):
        if self._in_th or self._in_td:
            self._current_cell += data


def _parse_table_html(html: str) -> tuple[list[str], list[list[str]]]:
    """Parse HTML table content into headers and rows.

    Args:
        html: HTML string containing a table.

    Returns:
        Tuple of (headers, rows).
    """
    parser = _TableHTMLParser()
    parser.feed(html)
    return parser.headers, parser.rows


def _map_block_type(reducto_type: str) -> str:
    """Map Reducto block type to our internal block type.

    Args:
        reducto_type: Block type string from Reducto API.

    Returns:
        Internal block type string.
    """
    type_lower = reducto_type.lower()
    if type_lower in ("title", "section header", "header"):
        return "heading"
    if type_lower == "list item":
        return "list_item"
    return "paragraph"


def _convert_bbox(bbox: dict) -> list[float]:
    """Convert Reducto bbox format to [x0, y0, x1, y1].

    Reducto uses {left, top, width, height} with normalized 0-1 values.

    Args:
        bbox: Reducto bbox dictionary.

    Returns:
        List of [x0, y0, x1, y1].
    """
    left = bbox.get("left", 0.0)
    top = bbox.get("top", 0.0)
    width = bbox.get("width", 0.0)
    height = bbox.get("height", 0.0)
    return [left, top, left + width, top + height]


class ReductoParser:
    """PDF parser using the Reducto cloud API.

    Initializes the Reducto client once and reuses it across parse calls.

    Args:
        api_key: Reducto API key. If None, reads from REDUCTO_API_KEY env var.

    Raises:
        ValueError: If no API key is provided or found in environment.
    """

    def __init__(self, api_key: str | None = None):
        if Reducto is None:
            raise ImportError(
                "reductoai package is required for ReductoParser. "
                "Install it with: pip install 'reductoai>=0.16.0'"
            )
        api_key = api_key or os.getenv("REDUCTO_API_KEY")
        if not api_key:
            raise ValueError(
                "REDUCTO_API_KEY environment variable is not set. "
                "Set it or pass api_key to use the Reducto parser."
            )
        self.client = Reducto(api_key=api_key)

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """Parse a PDF file using the Reducto cloud API.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument containing all extracted pages, blocks, and tables.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        start = time.perf_counter()
        logger.info("parsing pdf with reducto", file_path=str(file_path))

        upload = self.client.upload(file=file_path)
        result = self.client.parse.run(input=upload)

        # Group blocks by page number
        pages_dict: dict[int, dict] = {}

        for chunk in result.chunks:
            for block in chunk.blocks:
                page_num = block.bbox.page if block.bbox else 0
                if page_num not in pages_dict:
                    pages_dict[page_num] = {"blocks": [], "tables": []}

                block_type_str = str(block.block_type) if block.block_type else "paragraph"

                if block_type_str.lower() == "table":
                    # Parse table HTML content
                    content = block.content or ""
                    headers, rows = _parse_table_html(content)
                    table_index = len(pages_dict[page_num]["tables"])
                    pages_dict[page_num]["tables"].append(
                        TableData(
                            table_index=table_index,
                            headers=headers,
                            rows=rows,
                        )
                    )
                else:
                    mapped_type = _map_block_type(block_type_str)
                    bbox = _convert_bbox(block.bbox.__dict__) if block.bbox else None
                    block_index = len(pages_dict[page_num]["blocks"])
                    pages_dict[page_num]["blocks"].append(
                        TextBlock(
                            block_index=block_index,
                            block_type=mapped_type,
                            text=block.content or "",
                            font_size=12.0,
                            is_bold=mapped_type == "heading",
                            bbox=bbox,
                        )
                    )

        # Build sorted page list
        pages = []
        for page_num in sorted(pages_dict.keys()):
            page_data = pages_dict[page_num]
            pages.append(
                ParsedPage(
                    page_number=page_num,  # Reducto bbox.page is already 1-indexed
                    blocks=page_data["blocks"],
                    tables=page_data["tables"],
                )
            )

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "pdf parsed with reducto",
            file_path=str(file_path),
            total_pages=len(pages),
            total_blocks=sum(len(p.blocks) for p in pages),
            total_tables=sum(len(p.tables) for p in pages),
            duration_ms=round(duration_ms, 2),
        )

        return ParsedDocument(
            file_path=str(file_path),
            total_pages=len(pages),
            pages=pages,
        )
