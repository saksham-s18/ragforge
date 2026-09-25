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


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when a vector dimension does not match the expected store dimension."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when connecting to the vector store fails or times out."""


class EmbeddingError(RAGForgeError):
    """Raised when embedding generation fails."""


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when the specified embedding model cannot be found or loaded."""


class LLMError(RAGForgeError):
    """Base exception for all LLM provider and generation errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        is_fallback_eligible: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.is_fallback_eligible = is_fallback_eligible


class LLMConfigurationError(LLMError):
    """Raised when an LLM provider is improperly configured (e.g. missing API key)."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message, provider=provider, is_fallback_eligible=False)


class LLMAuthenticationError(LLMError):
    """Raised when provider authentication fails (e.g. invalid API key)."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message, provider=provider, is_fallback_eligible=False)


class LLMInvalidRequestError(LLMError):
    """Raised when request payload or parameters are invalid."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message, provider=provider, is_fallback_eligible=False)


class LLMTransientError(LLMError):
    """Base exception for transient/recoverable errors eligible for fallback."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message, provider=provider, is_fallback_eligible=True)


class LLMRateLimitError(LLMTransientError):
    """Raised when an LLM provider rate limit is exceeded."""


class LLMTimeoutError(LLMTransientError):
    """Raised when an LLM generation request times out."""


class LLMConnectionError(LLMTransientError):
    """Raised when network connection to an LLM provider fails."""


class LLMProviderUnavailableError(LLMTransientError):
    """Raised when an LLM provider service is overloaded or unavailable (5xx)."""


class AllLLMProvidersFailedError(LLMError):
    """Raised when both primary and fallback LLM providers fail."""

    def __init__(
        self,
        message: str | None = None,
        primary_provider: str | None = None,
        fallback_provider: str | None = None,
        primary_error: Exception | None = None,
        fallback_error: Exception | None = None,
    ) -> None:
        if not message:
            p_name = primary_provider or "primary"
            f_name = fallback_provider or "fallback"
            message = (
                f"All configured LLM providers failed: primary ({p_name}) failed with "
                f"[{primary_error}], fallback ({f_name}) failed with [{fallback_error}]."
            )
        super().__init__(message, provider=primary_provider, is_fallback_eligible=False)
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.primary_error = primary_error
        self.fallback_error = fallback_error


class RAGGenerationError(RAGForgeError):
    """Raised when RAG generation pipeline encounters an error."""
