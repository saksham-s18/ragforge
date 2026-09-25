"""Domain layer containing core data models, enums, and domain exceptions."""

from ragforge.domain.enums import DocumentStatus, IndexingStatus, MimeType, SearchStrategy
from ragforge.domain.exceptions import (
    ChunkingError,
    DocumentNotFoundError,
    EmbeddingError,
    EmbeddingModelNotFoundError,
    EmptyDocumentError,
    IngestionError,
    RAGForgeError,
    UnsupportedFileTypeError,
    VectorDimensionMismatchError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from ragforge.domain.models import (
    Chunk,
    ChunkMetadata,
    Citation,
    Document,
    DocumentIndexingResult,
    DocumentIndexRecord,
    IndexingResult,
    Query,
    RetrievedChunk,
)

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ChunkingError",
    "Citation",
    "Document",
    "DocumentIndexRecord",
    "DocumentIndexingResult",
    "DocumentNotFoundError",
    "DocumentStatus",
    "EmbeddingError",
    "EmbeddingModelNotFoundError",
    "EmptyDocumentError",
    "IndexingResult",
    "IndexingStatus",
    "IngestionError",
    "MimeType",
    "Query",
    "RAGForgeError",
    "RetrievedChunk",
    "SearchStrategy",
    "UnsupportedFileTypeError",
    "VectorDimensionMismatchError",
    "VectorStoreConnectionError",
    "VectorStoreError",
]
