from abc import ABC, abstractmethod

from ragforge.domain.models import DocumentIndexRecord


class BaseIndexStateStore(ABC):
    """Abstract port for persisting and retrieving indexing state and document provenance."""

    @abstractmethod
    async def get(self, file_path: str) -> DocumentIndexRecord | None:
        """Retrieve the indexing record for a file path, or None if not indexed."""

    @abstractmethod
    async def set(self, record: DocumentIndexRecord) -> None:
        """Store or update the indexing record for a file path."""

    @abstractmethod
    async def delete(self, file_path: str) -> bool:
        """Remove the indexing record for a file path. Returns True if removed."""

    @abstractmethod
    async def get_all(self) -> dict[str, DocumentIndexRecord]:
        """Return a mapping of all tracked file paths to their indexing records."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all indexing state records."""
