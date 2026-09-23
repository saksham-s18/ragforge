"""Domain layer containing core data models, enums, and domain exceptions."""

from ragforge.domain.enums import DocumentStatus, MimeType, SearchStrategy
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
    Query,
    RetrievedChunk,
)

__all__ = [
    "Chunk",
    "ChunkMetadata",
    "ChunkingError",
    "Citation",
    "Document",
    "DocumentNotFoundError",
    "DocumentStatus",
    "EmbeddingError",
    "EmbeddingModelNotFoundError",
    "EmptyDocumentError",
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
