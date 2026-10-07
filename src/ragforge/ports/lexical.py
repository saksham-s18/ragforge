from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from ragforge.domain.models import Chunk, RetrievedChunk


class BaseLexicalIndex(ABC):
    """Abstract port for lexical (BM25) search indexing and retrieval."""

    @abstractmethod
    async def index_chunks(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        """Index or update chunks in the lexical search index."""

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Execute lexical search for a query matching optional metadata filters."""

    @abstractmethod
    async def delete_by_document_id(
        self,
        document_id: UUID,
    ) -> None:
        """Delete all indexed chunks associated with a document ID."""

    @abstractmethod
    async def delete_by_chunk_id(
        self,
        chunk_id: UUID,
    ) -> bool:
        """Delete a single chunk by its ID. Returns True if found and deleted."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all indexed data from the lexical index."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of indexed chunks."""
