from ragforge.domain.models import DocumentIndexRecord
from ragforge.ports.state import BaseIndexStateStore


class InMemoryIndexStateStore(BaseIndexStateStore):
    """In-memory index state store adapter for testing, development, and ephemeral pipelines."""

    def __init__(self) -> None:
        self._records: dict[str, DocumentIndexRecord] = {}

    def count(self) -> int:
        """Return the count of tracked indexed document records."""
        return len(self._records)

    async def get(self, file_path: str) -> DocumentIndexRecord | None:
        """Retrieve the indexing record for a file path, or None if not indexed."""
        return self._records.get(file_path)

    async def set(self, record: DocumentIndexRecord) -> None:
        """Store or update the indexing record for a file path."""
        self._records[record.file_path] = record

    async def delete(self, file_path: str) -> bool:
        """Remove the indexing record for a file path. Returns True if removed."""
        if file_path in self._records:
            del self._records[file_path]
            return True
        return False

    async def get_all(self) -> dict[str, DocumentIndexRecord]:
        """Return a mapping of all tracked file paths to their indexing records."""
        return dict(self._records)

    async def clear(self) -> None:
        """Clear all indexing state records."""
        self._records.clear()
