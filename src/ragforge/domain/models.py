from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ragforge.domain.enums import DocumentStatus, IndexingStatus, MimeType, SearchStrategy


class Document(BaseModel):
    """Represents an ingested document entity."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    source_type: MimeType
    content_hash: str
    raw_content: str | None = None
    file_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChunkMetadata(BaseModel):
    """Granular positional and structural metadata for a chunk."""

    page_number: int | None = None
    section_header: str | None = None
    start_char_idx: int | None = None
    end_char_idx: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Represents a discrete text chunk extracted from a Document."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_index: int
    content: str
    token_count: int
    content_hash: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    dense_vector: list[float] | None = None
    sparse_vector: dict[int, float] | None = None


class Query(BaseModel):
    """Represents an incoming retrieval query with execution parameters."""

    id: UUID = Field(default_factory=uuid4)
    raw_query: str
    transformed_queries: list[str] = Field(default_factory=list)
    filters: dict[str, Any] | None = None
    top_k: int = 10
    score_threshold: float | None = None
    search_strategy: SearchStrategy = SearchStrategy.HYBRID


class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved from a vector store or ranking service."""

    chunk: Chunk
    score: float
    retrieval_type: str = "dense"
    rank: int


class Citation(BaseModel):
    """Direct attribution mapping from an answer back to a source chunk."""

    chunk_id: UUID
    document_id: UUID
    document_title: str
    page_number: int | None = None
    matched_text_snippet: str


class DocumentIndexRecord(BaseModel):
    """Record tracking the indexed state of a document for incremental indexing."""

    file_path: str
    document_id: UUID
    content_hash: str
    chunk_count: int = 0
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIndexingResult(BaseModel):
    """Outcome for an individual document during an indexing run."""

    file_path: str
    status: IndexingStatus
    document_id: UUID | None = None
    chunk_count: int = 0
    content_hash: str | None = None
    error: str | None = None


class IndexingResult(BaseModel):
    """Aggregate statistics and document outcomes for an indexing execution."""

    discovered_documents: int = 0
    indexed_documents: int = 0
    updated_documents: int = 0
    skipped_documents: int = 0
    failed_documents: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    duration_seconds: float = 0.0
    documents: list[DocumentIndexingResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total number of documents evaluated during the run."""
        return (
            self.indexed_documents
            + self.updated_documents
            + self.skipped_documents
            + self.failed_documents
        )

    @property
    def is_success(self) -> bool:
        """True if no document-level failures occurred during indexing."""
        return self.failed_documents == 0
