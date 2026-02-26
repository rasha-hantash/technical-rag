from .models import ChunkData, IngestedDocument, ChunkRecord, SearchResult
from .database import PgVectorStore
from .ingestion import (
    ParsedDocument,
    ParsedPage,
    TextBlock,
    TableData,
    parse_pdf,
    parse_pdf_pymupdf,
    ReductoParser,
    fixed_size_chunking,
    semantic_chunking_by_paragraphs,
    chunk_parsed_document,
    detect_content_type,
    assess_needs_ocr,
    ocr_pdf_with_tesseract,
    RAGIngestionPipeline,
    ingest_document,
    compute_file_hash,
    validate_file_path,
    IngestResult,
    PathValidationError,
)
from .llm_clients import (
    EmbeddingClient,
    EmbeddingResult,
    generate_embedding,
    generate_embeddings,
    AnthropicClient,
    CohereClient,
)
from .retrieval import (
    RAGRetriever,
    Reranker,
    CohereReranker,
    CrossEncoderReranker,
)
from .generation import (
    RAGGenerator,
    RAGResponse,
    SourceReference,
)

__all__ = [
    # Models
    "IngestedDocument",
    "ChunkRecord",
    "SearchResult",
    # Database
    "PgVectorStore",
    # PDF Parser
    "parse_pdf",
    "parse_pdf_pymupdf",
    "ReductoParser",
    "ParsedDocument",
    "ParsedPage",
    "TextBlock",
    "TableData",
    # Chunking
    "fixed_size_chunking",
    "semantic_chunking_by_paragraphs",
    "chunk_parsed_document",
    "detect_content_type",
    "ChunkData",
    # OCR
    "assess_needs_ocr",
    "ocr_pdf_with_tesseract",
    # Ingestion
    "RAGIngestionPipeline",
    "ingest_document",
    "compute_file_hash",
    "validate_file_path",
    "IngestResult",
    "PathValidationError",
    # Embeddings
    "EmbeddingClient",
    "EmbeddingResult",
    "generate_embedding",
    "generate_embeddings",
    # LLM Clients
    "AnthropicClient",
    "CohereClient",
    # Retriever
    "RAGRetriever",
    # Generation
    "RAGGenerator",
    "RAGResponse",
    "SourceReference",
    # Re-ranking
    "Reranker",
    "CohereReranker",
    "CrossEncoderReranker",
]
