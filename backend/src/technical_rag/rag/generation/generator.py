"""RAG response generator using Claude."""

import time
from uuid import UUID

from pydantic import BaseModel

from ...logger import logger
from ..llm_clients.anthropic_client import AnthropicClient
from ..models import SearchResult


class SourceReference(BaseModel):
    """A source reference from a retrieved chunk."""

    chunk_id: UUID | None = None
    document_id: UUID | None = None
    file_path: str
    page_number: int | None
    content: str
    content_preview: str  # First 200 chars of chunk
    bbox: list[float] | None = None  # [x0, y0, x1, y1] coordinates
    section_hierarchy: str | None = None
    book_title: str | None = None
    book_author: str | None = None
    publication_year: int | None = None
    tags: list[str] = []


class RAGResponse(BaseModel):
    """Response from a RAG query."""

    answer: str
    sources: list[SourceReference]
    chunks_used: int


class RAGGenerator:
    """Generates answers from retrieved search results using Claude."""

    DEFAULT_SYSTEM_PROMPT = """You are a technical architecture advisor that answers questions by referencing authoritative technical books.

Rules:
1. Only use information from the provided context to answer the question.
2. Always cite your sources using this format: [BookTitle, Chapter/Section, p.XX]
3. If multiple books discuss the same topic, compare their perspectives.
4. If a source's publication year suggests its advice may be outdated, explicitly note this:
   "Note: This advice is from [BookTitle] ([Year]). Practices in this area have evolved since then."
5. When the user provides code, evaluate it against the principles from the books.
   Be specific about which principle applies and why.
6. If the context doesn't contain enough information, say so clearly.
7. Be direct and technical in your responses."""

    def __init__(
        self,
        anthropic_client: AnthropicClient,
        system_prompt: str | None = None,
    ):
        """Initialize the RAG generator.

        Args:
            anthropic_client: AnthropicClient instance for Claude API calls.
            system_prompt: Custom system prompt for Claude. Uses default if not provided.
        """
        self.anthropic_client = anthropic_client
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

    def generate(self, question: str, results: list[SearchResult]) -> RAGResponse:
        """Generate an answer from search results.

        Args:
            question: The user's question.
            results: Search results from retrieval.

        Returns:
            RAGResponse with answer and source references.
        """
        start = time.perf_counter()

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                sources=[],
                chunks_used=0,
            )

        # Build context from results
        context = self._build_context(results)

        # Generate response with Claude
        user_message = f"""Context:
{context}

Question: {question}

Please answer the question based only on the provided context."""

        generation_start = time.perf_counter()
        answer = self.anthropic_client.create_message(
            system=self.system_prompt,
            user_message=user_message,
        )
        generation_duration_ms = (time.perf_counter() - generation_start) * 1000

        # Build source references
        sources = self._build_sources(results)

        total_duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "rag query completed",
            question_length=len(question),
            chunks_used=len(results),
            answer_length=len(answer),
            generation_duration_ms=round(generation_duration_ms, 2),
            total_duration_ms=round(total_duration_ms, 2),
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            chunks_used=len(results),
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        """Build context string from search results.

        Args:
            results: List of SearchResult objects.

        Returns:
            Formatted context string for the LLM.
        """
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            doc = result.document

            source_info = []
            if doc:
                book_label = doc.title or doc.file_path
                if doc.author:
                    book_label += f" by {doc.author}"
                if doc.publication_year:
                    book_label += f" ({doc.publication_year})"
                source_info.append(f"Book: {book_label}")
            if chunk.section_hierarchy:
                source_info.append(f"Section: {chunk.section_hierarchy}")
            if chunk.page_number is not None:
                source_info.append(f"Page: {chunk.page_number}")
            if chunk.chunk_type:
                source_info.append(f"Type: {chunk.chunk_type}")

            header = f"[Context {i}] " + " | ".join(source_info) if source_info else f"[Context {i}]"
            context_parts.append(f"{header}\n{chunk.content}")

        return "\n\n---\n\n".join(context_parts)

    def _build_sources(self, results: list[SearchResult]) -> list[SourceReference]:
        """Build source references from search results.

        Args:
            results: List of SearchResult objects.

        Returns:
            List of SourceReference objects.
        """
        sources = []
        for result in results:
            chunk = result.chunk
            doc = result.document

            sources.append(
                SourceReference(
                    chunk_id=chunk.id,
                    document_id=doc.id if doc else None,
                    file_path=doc.file_path if doc else "unknown",
                    page_number=chunk.page_number,
                    content=chunk.content,
                    content_preview=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                    bbox=chunk.bbox,
                    section_hierarchy=chunk.section_hierarchy,
                    book_title=doc.title if doc else None,
                    book_author=doc.author if doc else None,
                    publication_year=doc.publication_year if doc else None,
                    tags=doc.tags if doc else [],
                )
            )

        return sources
