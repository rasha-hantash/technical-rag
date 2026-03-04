"""FastAPI REST API for the RAG pipeline."""

import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field

from .logger import logger
from .rag import (
    AnthropicClient,
    CohereReranker,
    CrossEncoderReranker,
    EmbeddingClient,
    PathValidationError,
    PgVectorStore,
    RAGGenerator,
    RAGIngestionPipeline,
    RAGRetriever,
)

# Maximum file size for uploads (200MB for large technical books)
MAX_UPLOAD_SIZE = 200 * 1024 * 1024

# Maximum number of files in a single batch upload
MAX_BATCH_SIZE = 100

# Directory for persistent PDF storage
PDF_STORAGE_DIR = Path(os.getenv("PDF_STORAGE_DIR", "./data/pdfs"))


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)
    tags: list[str] | None = Field(default=None, description="Tag filter. None or omitted searches all books.")


class SourceResponse(BaseModel):
    chunk_id: UUID | None = None
    document_id: UUID | None = None
    file_path: str
    page_number: int | None
    content: str
    content_preview: str
    bbox: list[float] | None = None
    section_hierarchy: str | None = None
    book_title: str | None = None
    book_author: str | None = None
    publication_year: int | None = None
    tags: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    chunks_used: int


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)
    tags: list[str] | None = Field(default=None, description="Tag filter. None or omitted searches all books.")


class SearchResponse(BaseModel):
    sources: list[SourceResponse]
    chunks_retrieved: int


class BatchIngestItemResponse(BaseModel):
    file_name: str
    document_id: UUID | None = None
    chunks_count: int = 0
    was_duplicate: bool = False
    error: str | None = None


class BatchIngestResponse(BaseModel):
    results: list[BatchIngestItemResponse]
    successful: int = 0
    duplicates: int = 0
    failed: int = 0


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    code: str
    message: str


# --- Dependency Accessors ---


def get_db(request: Request) -> PgVectorStore:
    return request.app.state.db


def get_retriever(request: Request) -> RAGRetriever:
    return request.app.state.retriever


def get_generator(request: Request) -> RAGGenerator:
    return request.app.state.generator


def get_ingestion_pipeline(request: Request) -> RAGIngestionPipeline:
    return request.app.state.ingestion_pipeline


# --- Lifecycle ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services at startup, tear down on shutdown."""
    logger.info("starting server")

    PDF_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    app.state.db = PgVectorStore()
    app.state.db.connect()

    app.state.embedding_client = EmbeddingClient()

    app.state.reranker = None
    reranker_type = os.getenv("RERANKER", "").lower()
    if reranker_type == "cohere":
        app.state.reranker = CohereReranker()
    elif reranker_type == "cross-encoder":
        app.state.reranker = CrossEncoderReranker()

    app.state.retriever = RAGRetriever(
        db=app.state.db,
        embedding_client=app.state.embedding_client,
        reranker=app.state.reranker,
    )

    anthropic_client = AnthropicClient()
    app.state.generator = RAGGenerator(anthropic_client=anthropic_client)

    app.state.ingestion_pipeline = RAGIngestionPipeline(
        db=app.state.db,
        embedding_client=app.state.embedding_client,
    )

    logger.info("server ready")

    yield

    app.state.db.disconnect()
    logger.info("server shutdown")


app = FastAPI(
    title="Technical Book Rack API",
    description="Technical book ingestion and retrieval-augmented generation API",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Exception Handlers ---


@app.exception_handler(PathValidationError)
async def path_validation_handler(request, exc: PathValidationError):
    return JSONResponse(
        status_code=403,
        content=ErrorResponse(code="ACCESS_DENIED", message=str(exc)).model_dump(),
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(code="FILE_NOT_FOUND", message=str(exc)).model_dump(),
    )


# --- Health Endpoints ---


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check."""
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=HealthResponse)
def ready(db: PgVectorStore = Depends(get_db)):
    """Readiness check - verifies database connectivity."""
    checks = {"database": False}

    if db and db.conn:
        try:
            with db.conn.cursor() as cur:
                cur.execute("SELECT 1")
            checks["database"] = True
        except Exception as e:
            logger.debug("health check db query failed", error=str(e))

    status = "healthy" if all(checks.values()) else "unhealthy"
    return HealthResponse(status=status, checks=checks)


# --- RAG Endpoints ---


@app.post("/api/v1/rag/ingest/batch", response_model=BatchIngestResponse)
def ingest_batch(
    files: list[UploadFile] = File(...),
    tags: str = Form(..., description="JSON array of tags, e.g. '[\"rust\", \"backend\"]'. At least one tag required."),
    title: str | None = Form(None),
    author: str | None = Form(None),
    edition: str | None = Form(None),
    publication_year: int | None = Form(None, ge=1900, le=2030),
    pipeline: RAGIngestionPipeline = Depends(get_ingestion_pipeline),
):
    """Ingest multiple PDF files via batch upload."""
    try:
        parsed_tags = json.loads(tags)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="tags must be a valid JSON array")
    if not isinstance(parsed_tags, list) or not all(isinstance(t, str) for t in parsed_tags):
        raise HTTPException(status_code=400, detail="tags must be an array of strings")
    if len(parsed_tags) == 0:
        raise HTTPException(status_code=400, detail="At least one tag is required")

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size is {MAX_BATCH_SIZE}",
        )

    results: list[BatchIngestItemResponse] = []
    valid_tmp_paths: list[Path] = []
    valid_filenames: list[str] = []
    valid_file_sizes: list[int] = []
    all_tmp_paths: list[Path] = []

    # Phase 1: Validate each file and save to temp
    for file in files:
        file_name = file.filename or "unknown.pdf"

        if not file_name.lower().endswith(".pdf"):
            results.append(
                BatchIngestItemResponse(
                    file_name=file_name, error="Only PDF files are supported"
                )
            )
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(file.file, tmp)
        all_tmp_paths.append(tmp_path)

        actual_size = tmp_path.stat().st_size
        if actual_size > MAX_UPLOAD_SIZE:
            results.append(
                BatchIngestItemResponse(
                    file_name=file_name,
                    error=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)}MB",
                )
            )
            continue

        with open(tmp_path, "rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            results.append(
                BatchIngestItemResponse(
                    file_name=file_name,
                    error="Invalid PDF file. File does not have valid PDF header.",
                )
            )
            continue

        valid_tmp_paths.append(tmp_path)
        valid_filenames.append(file_name)
        valid_file_sizes.append(actual_size)

    # Phase 2: Batch ingest valid files
    try:
        if valid_tmp_paths:
            ingest_results = pipeline.ingest_batch(
                file_paths=valid_tmp_paths,
                original_filenames=valid_filenames,
                file_sizes=valid_file_sizes,
                title=title,
                author=author,
                edition=edition,
                publication_year=publication_year,
                tags=parsed_tags,
            )

            for i, result in enumerate(ingest_results):
                file_name = valid_filenames[i]
                if result.error:
                    results.append(
                        BatchIngestItemResponse(
                            file_name=file_name, error=result.error
                        )
                    )
                else:
                    if result.document and not result.was_duplicate:
                        pdf_dest = PDF_STORAGE_DIR / f"{result.document.id}.pdf"
                        shutil.copy2(valid_tmp_paths[i], pdf_dest)

                    results.append(
                        BatchIngestItemResponse(
                            file_name=file_name,
                            document_id=result.document.id
                            if result.document
                            else None,
                            chunks_count=result.chunks_count,
                            was_duplicate=result.was_duplicate,
                        )
                    )
    finally:
        for tmp_path in all_tmp_paths:
            tmp_path.unlink(missing_ok=True)

    successful = sum(
        1
        for r in results
        if r.document_id and not r.was_duplicate and not r.error
    )
    duplicates = sum(1 for r in results if r.was_duplicate)
    failed = sum(1 for r in results if r.error)

    return BatchIngestResponse(
        results=results,
        successful=successful,
        duplicates=duplicates,
        failed=failed,
    )


@app.post("/api/v1/rag/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    retriever: RAGRetriever = Depends(get_retriever),
    generator: RAGGenerator = Depends(get_generator),
):
    """Answer a question using RAG."""
    results = retriever.retrieve(request.question, top_k=request.top_k, tags=request.tags)
    response = generator.generate(request.question, results)
    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceResponse(
                chunk_id=s.chunk_id,
                document_id=s.document_id,
                file_path=s.file_path,
                page_number=s.page_number,
                content=s.content,
                content_preview=s.content_preview,
                bbox=s.bbox,
                section_hierarchy=s.section_hierarchy,
                book_title=s.book_title,
                book_author=s.book_author,
                publication_year=s.publication_year,
                tags=s.tags,
            )
            for s in response.sources
        ],
        chunks_used=response.chunks_used,
    )


@app.post("/api/v1/rag/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    retriever: RAGRetriever = Depends(get_retriever),
):
    """Search for relevant chunks without generating an answer (retrieval only)."""
    try:
        results = retriever.retrieve(request.question, top_k=request.top_k, tags=request.tags)
    except Exception as e:
        logger.error("search retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {type(e).__name__}")
    return SearchResponse(
        sources=[
            SourceResponse(
                chunk_id=r.chunk.id,
                document_id=r.document.id if r.document else None,
                file_path=r.document.file_path if r.document else "",
                page_number=r.chunk.page_number,
                content=r.chunk.content,
                content_preview=r.chunk.content[:200] if r.chunk.content else "",
                bbox=r.chunk.bbox,
                section_hierarchy=r.chunk.section_hierarchy,
                book_title=r.document.title if r.document else None,
                book_author=r.document.author if r.document else None,
                publication_year=r.document.publication_year if r.document else None,
                tags=r.document.tags if r.document else [],
            )
            for r in results
        ],
        chunks_retrieved=len(results),
    )


# --- Document Endpoints ---


class DocumentResponse(BaseModel):
    id: UUID
    file_path: str
    chunks_count: int
    status: str
    file_size: int | None = None
    title: str | None = None
    author: str | None = None
    edition: str | None = None
    publication_year: int | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: str


@app.get("/api/v1/documents", response_model=list[DocumentResponse])
def list_documents(db: PgVectorStore = Depends(get_db)):
    """List all ingested documents with chunk counts."""
    with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """SELECT d.id, d.file_path, d.status, d.file_size, d.title, d.author,
                      d.edition, d.publication_year, d.tags, d.created_at,
                      COUNT(c.id) AS chunks_count
               FROM documents d
               LEFT JOIN chunks c ON c.document_id = d.id
               GROUP BY d.id
               ORDER BY d.created_at DESC"""
        )
        rows = cur.fetchall()

    return [
        DocumentResponse(
            id=row["id"],
            file_path=row["file_path"],
            chunks_count=row["chunks_count"],
            status=row["status"],
            file_size=row["file_size"],
            title=row.get("title"),
            author=row.get("author"),
            edition=row.get("edition"),
            publication_year=row.get("publication_year"),
            tags=row.get("tags", []),
            created_at=row["created_at"].isoformat(),
        )
        for row in rows
    ]


@app.get("/api/v1/documents/{document_id}/file")
def get_document_file(document_id: UUID):
    """Serve the original PDF file for a document."""
    pdf_path = PDF_STORAGE_DIR / f"{document_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{document_id}.pdf",
    )


class BookMetadataRequest(BaseModel):
    title: str | None = None
    author: str | None = None
    edition: str | None = None
    publication_year: int | None = Field(default=None, ge=1900, le=2030)


class BookSectionResponse(BaseModel):
    section_hierarchy: str
    chunk_count: int
    start_page: int | None


@app.put("/api/v1/documents/{document_id}/metadata", response_model=DocumentResponse)
def update_document_metadata(
    document_id: UUID,
    body: BookMetadataRequest,
    db: PgVectorStore = Depends(get_db),
):
    """Update book metadata for a document."""
    db.update_book_metadata(
        document_id=document_id,
        title=body.title,
        author=body.author,
        edition=body.edition,
        publication_year=body.publication_year,
    )
    # Re-fetch to return updated document
    with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """SELECT d.id, d.file_path, d.status, d.file_size, d.title, d.author,
                      d.edition, d.publication_year, d.tags, d.created_at,
                      COUNT(c.id) AS chunks_count
               FROM documents d
               LEFT JOIN chunks c ON c.document_id = d.id
               WHERE d.id = %s
               GROUP BY d.id""",
            (str(document_id),),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(
        id=row["id"],
        file_path=row["file_path"],
        chunks_count=row["chunks_count"],
        status=row["status"],
        file_size=row["file_size"],
        title=row.get("title"),
        author=row.get("author"),
        edition=row.get("edition"),
        publication_year=row.get("publication_year"),
        tags=row.get("tags", []),
        created_at=row["created_at"].isoformat(),
    )


@app.get("/api/v1/documents/{document_id}/sections", response_model=list[BookSectionResponse])
def get_document_sections(
    document_id: UUID,
    db: PgVectorStore = Depends(get_db),
):
    """Get the chapter/section structure of a book."""
    rows = db.get_book_sections(document_id)
    return [
        BookSectionResponse(
            section_hierarchy=row["section_hierarchy"],
            chunk_count=row["chunk_count"],
            start_page=row.get("start_page"),
        )
        for row in rows
    ]
