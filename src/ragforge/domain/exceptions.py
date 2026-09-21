class RAGForgeError(Exception):
    """Base exception for all domain-specific errors in RAGForge."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DocumentNotFoundError(RAGForgeError):
    """Raised when a requested document cannot be located."""


class IngestionError(RAGForgeError):
    """Raised when document ingestion or parsing fails."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a document source has an unsupported MIME type or file extension."""


class EmptyDocumentError(IngestionError):
    """Raised when an ingested document contains no readable content."""


class ChunkingError(RAGForgeError):
    """Raised when chunking fails or invalid chunker parameters are supplied."""


class VectorStoreError(RAGForgeError):
    """Raised when a vector store operation fails."""
