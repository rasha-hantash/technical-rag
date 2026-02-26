"""Pipeline variant configuration for matrix evaluation.

Each variant is a combination of parser, chunking strategy, and reranker.
The matrix runner iterates over variants to compare end-to-end RAG quality.
"""

from pydantic import BaseModel


class PipelineVariant(BaseModel):
    """A single pipeline configuration to evaluate."""

    name: str
    parser: str  # "pymupdf" or "reducto"
    chunking: str  # "semantic" or "fixed"
    reranker: str  # "none", "cohere", or "cross-encoder"

    @property
    def label(self) -> str:
        return f"{self.parser}-{self.chunking}-{self.reranker}"


def get_default_matrix() -> list[PipelineVariant]:
    """Return the full evaluation matrix (12 variants)."""
    variants = []
    for parser in ("pymupdf", "reducto"):
        for chunking in ("semantic", "fixed"):
            for reranker in ("none", "cohere", "cross-encoder"):
                name = f"{parser}-{chunking}-{reranker}"
                variants.append(
                    PipelineVariant(
                        name=name,
                        parser=parser,
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
            name="reducto-semantic-none",
            parser="reducto",
            chunking="semantic",
            reranker="none",
        ),
    ]
