"""MCP server that wraps the Technical RAG FastAPI backend."""

import os

import httpx
from mcp.server.fastmcp import FastMCP

BACKEND_URL = os.getenv("TECHNICAL_RAG_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
MAX_CHUNK_CHARS = 1500  # Truncate long chunks to protect context budget

mcp = FastMCP(
    "technical-rag",
    instructions=(
        "Search through indexed technical books (programming, systems, etc.) "
        "using semantic search. Use `search` to find relevant passages, "
        "`list_documents` to see what books are indexed, and `browse_sections` "
        "to explore a book's chapter/section structure."
    ),
)


def _make_client() -> httpx.Client:
    return httpx.Client(base_url=BACKEND_URL, timeout=TIMEOUT)


def _format_source(src: dict, idx: int) -> str:
    """Format a single source into a readable string."""
    parts = [f"### Result {idx}"]

    title = src.get("book_title")
    author = src.get("book_author")
    if title:
        line = f"**{title}**"
        if author:
            line += f" by {author}"
        parts.append(line)

    section = src.get("section_hierarchy")
    page = src.get("page_number")
    location_parts = []
    if section:
        location_parts.append(section)
    if page is not None:
        location_parts.append(f"p. {page}")
    if location_parts:
        parts.append(" | ".join(location_parts))

    content = src.get("content", "").strip()
    if content:
        if len(content) > MAX_CHUNK_CHARS:
            content = content[:MAX_CHUNK_CHARS] + "... [truncated]"
        parts.append(f"\n{content}")

    return "\n".join(parts)


@mcp.tool()
def search(question: str, top_k: int = 5, tags: list[str] | None = None) -> str:
    """Search indexed technical books for passages relevant to your question.

    Uses semantic search (embedding similarity + BM25) with optional reranking
    to find the most relevant chunks across all indexed books.

    Args:
        question: The question or topic to search for.
        top_k: Number of results to return (1-20, default 5).
        tags: Optional list of tags to filter by (e.g. ["rust", "networking"]).
              If omitted, searches all books.

    Returns:
        Formatted search results with content, book title, section, and page number.
    """
    if not 1 <= top_k <= 20:
        return "Error: top_k must be between 1 and 20."

    payload: dict = {"question": question, "top_k": top_k}
    if tags:
        payload["tags"] = tags

    try:
        with _make_client() as client:
            resp = client.post("/api/v1/rag/search", json=payload)
            resp.raise_for_status()
    except httpx.ConnectError:
        return f"Error: Could not connect to the RAG backend at {BACKEND_URL}. Is it running? (start with: cd technical-rag/backend && uv run python main.py)"
    except httpx.ReadTimeout:
        return "Error: Backend timed out. The query may be too broad — try a more specific question."
    except httpx.HTTPStatusError as e:
        return f"Error: Backend returned HTTP {e.response.status_code}. Check backend logs at /tmp/technical-rag-backend.log"
    except Exception as e:
        return f"Error: Unexpected error communicating with RAG backend: {type(e).__name__}: {e}"

    data = resp.json()
    sources = data.get("sources", [])

    if not sources:
        return f"No results found for: {question}"

    formatted = [f"Found {len(sources)} results for: {question}\n"]
    for i, src in enumerate(sources, 1):
        formatted.append(_format_source(src, i))

    return "\n\n".join(formatted)


@mcp.tool()
def list_documents() -> str:
    """List all indexed books/documents in the RAG system.

    Returns:
        A formatted list of all documents with their title, author, tags, and chunk count.
    """
    try:
        with _make_client() as client:
            resp = client.get("/api/v1/documents")
            resp.raise_for_status()
    except httpx.ConnectError:
        return f"Error: Could not connect to the RAG backend at {BACKEND_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        return f"Error: Backend returned HTTP {e.response.status_code}."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    docs = resp.json()

    if not docs:
        return "No documents indexed yet."

    lines = [f"**{len(docs)} document(s) indexed:**\n"]
    for doc in docs:
        title = doc.get("title") or doc.get("file_path", "Unknown")
        author = doc.get("author")
        chunks = doc.get("chunks_count", 0)
        tags = doc.get("tags", [])
        doc_id = doc.get("id", "")

        line = f"- **{title}**"
        if author:
            line += f" by {author}"
        line += f" ({chunks} chunks)"
        if tags:
            line += f" [{', '.join(tags)}]"
        line += f"\n  ID: `{doc_id}`"
        lines.append(line)

    return "\n".join(lines)


@mcp.tool()
def browse_sections(document_id: str) -> str:
    """Browse the chapter/section structure of an indexed book.

    Use `list_documents` first to get the document ID.

    Args:
        document_id: UUID of the document to browse.

    Returns:
        A formatted tree of sections with chunk counts and starting page numbers.
    """
    try:
        with _make_client() as client:
            resp = client.get(f"/api/v1/documents/{document_id}/sections")
            resp.raise_for_status()
    except httpx.ConnectError:
        return f"Error: Could not connect to the RAG backend at {BACKEND_URL}. Is it running?"
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return f"Document not found: {document_id}"
        return f"Error: Backend returned HTTP {e.response.status_code}."
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    sections = resp.json()

    if not sections:
        return f"No sections found for document {document_id}."

    lines = [f"**{len(sections)} section(s):**\n"]
    for sec in sections:
        hierarchy = sec.get("section_hierarchy", "Unknown")
        chunk_count = sec.get("chunk_count", 0)
        start_page = sec.get("start_page")

        line = f"- {hierarchy} ({chunk_count} chunks"
        if start_page is not None:
            line += f", starts p. {start_page}"
        line += ")"
        lines.append(line)

    return "\n".join(lines)


def main():
    mcp.run(transport="stdio")
