"""Pipeline variant configuration for matrix evaluation.

Each variant is a combination of parser, chunking strategy, and reranker.
The matrix runner iterates over variants to compare end-to-end RAG quality.
"""

from pydantic import BaseModel


class PipelineVariant(BaseModel):
    """A single pipeline configuration to evaluate."""

    name: str
    parser: str  # "pymupdf"
    chunking: str  # "semantic" or "fixed"
    reranker: str  # "none", "cohere", or "cross-encoder"

    @property
    def label(self) -> str:
        return f"{self.parser}-{self.chunking}-{self.reranker}"


def get_default_matrix() -> list[PipelineVariant]:
    """Return the full evaluation matrix (6 variants)."""
    variants = []
    for chunking in ("semantic", "fixed"):
        for reranker in ("none", "cohere", "cross-encoder"):
            name = f"pymupdf-{chunking}-{reranker}"
            variants.append(
                PipelineVariant(
                    name=name,
                    parser="pymupdf",
                    chunking=chunking,
                    reranker=reranker,
                )
            )
    return variants


def get_quick_matrix() -> list[PipelineVariant]:
    """Return a minimal matrix for fast iteration (2 variants)."""
    return [
        PipelineVariant(
            name="pymupdf-semantic-none",
            parser="pymupdf",
            chunking="semantic",
            reranker="none",
        ),
        PipelineVariant(
            name="pymupdf-semantic-cohere",
            parser="pymupdf",
            chunking="semantic",
            reranker="cohere",
        ),
    ]
