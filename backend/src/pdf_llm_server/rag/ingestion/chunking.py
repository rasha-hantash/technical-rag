"""Text chunking utilities for the RAG pipeline."""

import re

from ..models import ChunkData
from .parser_models import ParsedDocument


def fixed_size_chunking(
    text: str, chunk_size: int = 1000, overlap: int = 200
) -> list[str]:
    """Split text into fixed-size chunks with overlap.

    Args:
        text: The text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text or chunk_size <= 0:
        return []

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size to avoid infinite loops")

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at word boundary
        if end < len(text):
            # Look for last space within chunk
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunks.append(text[start:end].strip())

        # Break after processing the final chunk
        if end >= len(text):
            break

        start = end - overlap

    return [c for c in chunks if c]


def semantic_chunking_by_paragraphs(
    text: str, max_chunk_size: int = 1500
) -> list[str]:
    """Split text by paragraphs, merging small paragraphs.

    Args:
        text: The text to chunk.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        List of text chunks preserving paragraph boundaries.
    """
    if not text:
        return []

    # Split by double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # If single paragraph exceeds max, use fixed-size chunking
        if para_size > max_chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            # Chunk the large paragraph
            chunks.extend(fixed_size_chunking(para, max_chunk_size, overlap=200))
            continue

        # Check if adding this paragraph exceeds max
        new_size = current_size + para_size + (2 if current_chunk else 0)
        if new_size > max_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(para)
        current_size += para_size + (2 if len(current_chunk) > 1 else 0)

    # Flush remaining
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def detect_content_type(text: str) -> str:
    """Detect the type of content in a text block.

    Args:
        text: The text to classify.

    Returns:
        Content type: "heading", "paragraph", "list", or "table".
    """
    if not text:
        return "paragraph"

    stripped = text.strip()
    lines = stripped.split("\n")

    # Check for table-like content (multiple | characters)
    if "|" in stripped and lines[0].count("|") >= 2:
        return "table"

    # Check for list items
    list_patterns = [
        r"^[\s]*[•◦▪▸►\-\*]",  # Bullet patterns
        r"^[\s]*\d+[\.\)]",  # Numbered list
        r"^[\s]*[a-zA-Z][\.\)]",  # Lettered list
    ]
    list_count = sum(
        1 for line in lines if any(re.match(p, line) for p in list_patterns)
    )
    if list_count > len(lines) / 2:
        return "list"

    # Short uppercase text is likely a heading
    if len(stripped) < 100 and stripped.isupper():
        return "heading"

    # Short text with no punctuation at end might be heading
    if len(stripped) < 80 and not stripped.endswith((".", "!", "?", ":")):
        return "heading"

    return "paragraph"


def _find_chunk_block_bboxes(
    chunk_text: str,
    block_list: list[tuple[str, list[float] | None]],
) -> list[list[float]]:
    """Find which blocks contributed text to a chunk and return their bboxes.

    Walks through the joined text to find the character range of the chunk,
    then returns bboxes for all blocks that overlap that range.
    """
    joined = " ".join(text for text, _ in block_list)
    chunk_start = joined.find(chunk_text)
    if chunk_start == -1:
        # Fallback: return all non-None bboxes
        return [list(bbox) for _, bbox in block_list if bbox is not None]

    chunk_end = chunk_start + len(chunk_text)
    bboxes = []
    pos = 0

    for text, bbox in block_list:
        block_start = pos
        block_end = pos + len(text)

        if block_end > chunk_start and block_start < chunk_end:
            if bbox is not None:
                bboxes.append(list(bbox))

        pos = block_end + 1  # +1 for the space separator

    return bboxes


def chunk_parsed_document(
    doc: ParsedDocument, strategy: str = "semantic"
) -> list[ChunkData]:
    """Chunk a parsed PDF document.

    Args:
        doc: ParsedDocument from parse_pdf().
        strategy: "semantic" for paragraph-aware or "fixed" for fixed-size.

    Returns:
        List of ChunkData ready for database insertion.
    """
    chunks = []
    position = 0

    for page in doc.pages:
        # Group consecutive blocks of same type, tracking per-block info
        groups: list[tuple[str, list[tuple[str, list[float] | None]]]] = []
        current_type: str | None = None
        current_blocks: list[tuple[str, list[float] | None]] = []

        for block in page.blocks:
            is_heading = block.block_type == "heading"
            was_heading = current_type == "heading"

            # Flush when crossing heading <-> non-heading boundary
            if current_blocks and (is_heading != was_heading):
                groups.append((current_type, current_blocks))
                current_blocks = []
                current_type = None

            if current_type is None:
                current_type = block.block_type
            current_blocks.append((block.text, block.bbox))

        if current_blocks:
            groups.append((current_type, current_blocks))

        # Apply chunking strategy to each group
        for block_type, block_list in groups:
            combined_text = " ".join(text for text, _ in block_list)

            if strategy == "fixed":
                text_chunks = fixed_size_chunking(combined_text)
            else:  # semantic
                text_chunks = semantic_chunking_by_paragraphs(combined_text)

            for chunk_text in text_chunks:
                if not chunk_text.strip():
                    continue

                # Find which blocks contributed to this chunk
                block_bboxes = _find_chunk_block_bboxes(
                    chunk_text, block_list
                )
                first_bbox = block_bboxes[0] if block_bboxes else None

                chunks.append(
                    ChunkData(
                        content=chunk_text,
                        chunk_type=block_type or "paragraph",
                        page_number=page.page_number,
                        position=position,
                        bbox=first_bbox,
                        block_bboxes=block_bboxes or None,
                    )
                )
                position += 1

        # Handle tables as separate chunks
        for table in page.tables:
            # Convert table to text representation
            table_lines = []
            if table.headers:
                table_lines.append(" | ".join(table.headers))
                table_lines.append("-" * 40)
            for row in table.rows:
                table_lines.append(" | ".join(row))

            if table_lines:
                chunks.append(
                    ChunkData(
                        content="\n".join(table_lines),
                        chunk_type="table",
                        page_number=page.page_number,
                        position=position,
                        bbox=None,
                    )
                )
                position += 1

    return chunks
