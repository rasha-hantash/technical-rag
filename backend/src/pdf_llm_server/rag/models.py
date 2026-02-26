from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from ..logger import logger


class IngestedDocument(BaseModel):
    id: UUID
    file_hash: str
    file_path: str
    metadata: dict = Field(default_factory=dict)
    status: str = "processing"
    file_size: int | None = None
    created_at: datetime


class ChunkRecord(BaseModel):
    id: UUID | None = None
    document_id: UUID
    content: str
    chunk_type: str | None = None
    page_number: int | None = None
    position: int | None = None
    embedding: list[float] | None = None
    bbox: list[float] | None = None  # [x0, y0, x1, y1] coordinates
    created_at: datetime | None = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != 4:
            raise ValueError("bbox must contain exactly 4 coordinates [x0, y0, x1, y1]")
        return v


class ChunkData(BaseModel):
    """A chunk of content ready for embedding."""

    content: str

    @field_validator("content")
    @classmethod
    def strip_nul_bytes(cls, v: str) -> str:
        """Remove NUL (0x00) characters that PostgreSQL cannot store in text fields."""
        if "\x00" not in v:
            return v
        logger.debug("stripped nul bytes from chunk content", original_length=len(v))
        return v.replace("\x00", "")

    chunk_type: str
    page_number: int
    position: int
    bbox: list[float] | None = None
    block_bboxes: list[list[float]] | None = None
    embedding: list[float] | None = None


class SearchResult(BaseModel):
    chunk: ChunkRecord
    score: float
    document: IngestedDocument | None = None
