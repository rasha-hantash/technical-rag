"""Ground truth schema and persistence for PDF extraction evaluation.

Defines the annotation data model used by the interactive HTML report
and consumed by the scoring engine.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Assessment of an extracted block's quality."""

    correct = "correct"
    partial = "partial"
    wrong = "wrong"


class BlockVerdict(BaseModel):
    """Annotation for a single extracted block."""

    block_index: int
    block_type: str
    verdict: Verdict
    original_text: str
    corrected_text: str | None = None
    notes: str | None = None


class MissingBlock(BaseModel):
    """Content the parser missed entirely."""

    block_type: str
    expected_text: str
    approximate_position: int | None = None
    notes: str | None = None


class PageAnnotation(BaseModel):
    """All annotations for a single page."""

    page_number: int
    block_verdicts: list[BlockVerdict] = Field(default_factory=list)
    missing_blocks: list[MissingBlock] = Field(default_factory=list)
    page_notes: str | None = None


class GroundTruth(BaseModel):
    """Top-level ground truth document for a single PDF + parser combination."""

    file_name: str
    file_hash: str
    parser_name: str
    annotator: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""
    pages: list[PageAnnotation] = Field(default_factory=list)


# Ground truth files for a single document should never approach this size.
_MAX_GROUND_TRUTH_BYTES = 50 * 1024 * 1024  # 50 MB


def save_ground_truth(gt: GroundTruth, path: str | Path) -> None:
    """Write ground truth to a JSON file."""
    path = Path(path).resolve()
    path.write_text(gt.model_dump_json(indent=2))


def load_ground_truth(path: str | Path) -> GroundTruth:
    """Load ground truth from a JSON file."""
    path = Path(path).resolve()
    size = path.stat().st_size
    if size > _MAX_GROUND_TRUTH_BYTES:
        raise ValueError(
            f"Ground truth file is {size / 1024 / 1024:.1f} MB, "
            f"exceeds {_MAX_GROUND_TRUTH_BYTES / 1024 / 1024:.0f} MB limit"
        )
    data = json.loads(path.read_text())
    return GroundTruth.model_validate(data)
