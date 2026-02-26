"""Ground truth schema and persistence for RAG pipeline evaluation.

Defines the dataset model used by the matrix evaluation runner:
questions with gold passages and expected answers over a PDF corpus.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    """Category of an evaluation question."""

    settlement = "settlement"
    eligibility = "eligibility"
    parties = "parties"
    deadlines = "deadlines"
    claim_calculation = "claim_calculation"
    multi_doc = "multi_doc"
    table = "table"


class GoldPassage(BaseModel):
    """A passage from a source PDF that contains (part of) the answer.

    Text is stored as a snippet, not a chunk ID, so it works across all
    parser/chunking combinations.
    """

    text: str
    source_pdf: str
    page_number: int | None = None


class ExpectedAnswer(BaseModel):
    """The expected answer for an evaluation question."""

    answer_text: str
    key_facts: list[str] = Field(default_factory=list)


class EvalQuestion(BaseModel):
    """A single evaluation question with ground truth."""

    question: str
    gold_passages: list[GoldPassage] = Field(default_factory=list)
    expected_answer: ExpectedAnswer
    category: QuestionCategory


class EvalDataset(BaseModel):
    """Top-level evaluation dataset over a PDF corpus."""

    name: str = ""
    description: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    pdf_corpus: list[str] = Field(default_factory=list)
    questions: list[EvalQuestion] = Field(default_factory=list)


_MAX_DATASET_BYTES = 50 * 1024 * 1024  # 50 MB


def save_eval_dataset(dataset: EvalDataset, path: str | Path) -> None:
    """Write an evaluation dataset to a JSON file."""
    path = Path(path).resolve()
    path.write_text(dataset.model_dump_json(indent=2))


def load_eval_dataset(path: str | Path) -> EvalDataset:
    """Load an evaluation dataset from a JSON file."""
    path = Path(path).resolve()
    size = path.stat().st_size
    if size > _MAX_DATASET_BYTES:
        raise ValueError(
            f"Dataset file is {size / 1024 / 1024:.1f} MB, "
            f"exceeds {_MAX_DATASET_BYTES / 1024 / 1024:.0f} MB limit"
        )
    data = json.loads(path.read_text())
    return EvalDataset.model_validate(data)
