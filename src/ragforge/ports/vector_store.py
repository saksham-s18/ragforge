from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from ragforge.domain.models import Chunk, RetrievedChunk


class BaseVectorStore(ABC):
    """Abstract port for vector storage and similarity retrieval."""

    @abstractmethod
    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Store or update chunks with their vector representations."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k nearest chunks matching the query vector and optional filters."""

    @abstractmethod
    async def delete_by_document_id(self, document_id: UUID) -> None:
        """Delete all indexed chunks associated with a document ID."""
