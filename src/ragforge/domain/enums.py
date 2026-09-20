from enum import StrEnum


class DocumentStatus(StrEnum):
    """Lifecycle status of a document during ingestion and indexing."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class MimeType(StrEnum):
    """Supported document MIME types."""

    PDF = "application/pdf"
    TXT = "text/plain"
    MARKDOWN = "text/markdown"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class SearchStrategy(StrEnum):
    """Retrieval search strategies."""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
