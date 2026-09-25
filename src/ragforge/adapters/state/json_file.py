import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ragforge.domain.models import DocumentIndexRecord
from ragforge.ports.state import BaseIndexStateStore

logger = logging.getLogger(__name__)


class JsonFileIndexStateStore(BaseIndexStateStore):
    """File-backed index state store persisting document indexing provenance to JSON.

    Provides durable state persistence across CLI invocations and application restarts.
    Uses atomic file replacement to prevent state corruption in case of unexpected termination.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path).resolve()
        self._loaded = False
        self._records: dict[str, DocumentIndexRecord] = {}

    def _ensure_loaded(self) -> None:
        """Load records from disk if not already loaded."""
        if self._loaded:
            return

        if self.file_path.exists():
            try:
                content = self.file_path.read_text(encoding="utf-8").strip()
                if content:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        self._records = {
                            path: DocumentIndexRecord.model_validate(rec)
                            for path, rec in data.items()
                        }
            except Exception as exc:
                logger.warning(
                    "Failed to read index state from '%s': %s. Initializing empty state.",
                    self.file_path,
                    exc,
                )
                self._records = {}

        self._loaded = True

    def _save_to_disk(self) -> None:
        """Atomically persist records dictionary to JSON file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_data = {path: rec.model_dump(mode="json") for path, rec in self._records.items()}
        json_text = json.dumps(serialized_data, indent=2, sort_keys=True)

        # Atomic write via temporary file in same directory
        temp_file = NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.file_path.parent,
            delete=False,
        )
        try:
            temp_file.write(json_text)
            temp_file.flush()
            temp_file.close()
            os.replace(temp_file.name, self.file_path)
        except Exception:
            if os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except OSError:
                    pass
            raise

    def count(self) -> int:
        """Return the count of tracked indexed document records."""
        self._ensure_loaded()
        return len(self._records)

    async def get(self, file_path: str) -> DocumentIndexRecord | None:
        """Retrieve the indexing record for a file path, or None if not indexed."""
        self._ensure_loaded()
        return self._records.get(file_path)

    async def set(self, record: DocumentIndexRecord) -> None:
        """Store or update the indexing record for a file path and persist."""
        self._ensure_loaded()
        self._records[record.file_path] = record
        self._save_to_disk()

    async def delete(self, file_path: str) -> bool:
        """Remove the indexing record for a file path and persist. Returns True if removed."""
        self._ensure_loaded()
        if file_path in self._records:
            del self._records[file_path]
            self._save_to_disk()
            return True
        return False

    async def get_all(self) -> dict[str, DocumentIndexRecord]:
        """Return a mapping of all tracked file paths to their indexing records."""
        self._ensure_loaded()
        return dict(self._records)

    async def clear(self) -> None:
        """Clear all indexing state records and persist empty state."""
        self._ensure_loaded()
        self._records.clear()
        self._save_to_disk()
