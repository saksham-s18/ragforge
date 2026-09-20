class RAGForgeError(Exception):
    """Base exception for all domain-specific errors in RAGForge."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DocumentNotFoundError(RAGForgeError):
    """Raised when a requested document cannot be located."""


class IngestionError(RAGForgeError):
    """Raised when document ingestion or parsing fails."""


class VectorStoreError(RAGForgeError):
    """Raised when a vector store operation fails."""
