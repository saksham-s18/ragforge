import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from ragforge.domain.enums import DocumentStatus, MimeType
from ragforge.domain.exceptions import (
    DocumentNotFoundError,
    EmptyDocumentError,
    IngestionError,
    UnsupportedFileTypeError,
)
from ragforge.domain.models import Document
from ragforge.ports.loaders import BaseDocumentLoader

# Deterministic namespace for RAGForge document identity generation
RAGFORGE_DOC_NAMESPACE = UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def generate_document_id(content_hash: str) -> UUID:
    """Generate a deterministic UUIDv5 identifier for a document based on content hash."""
    return uuid5(RAGFORGE_DOC_NAMESPACE, f"urn:ragforge:doc:{content_hash}")


class BaseFileLoader(BaseDocumentLoader, ABC):
    """Abstract base adapter for reading local text-based files into domain Documents."""

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Set of lowercase file extensions supported by this loader (e.g. {'.txt'})."""

    @property
    @abstractmethod
    def mime_type(self) -> MimeType:
        """Target domain MimeType for documents produced by this loader."""

    def _extract_title(self, path: Path, content: str) -> str:
        """Extract or derive a human-readable title for the document."""
        return path.name

    async def load(self, source: str | Path, allow_empty: bool = False, **kwargs: Any) -> Document:
        """Parse source file into a domain Document entity.

        Args:
            source: Local file path as string or Path object.
            allow_empty: If False, raises EmptyDocumentError when file has no content.
            **kwargs: Optional overrides such as 'title', 'document_id', or extra 'metadata'.

        Returns:
            A fully populated domain Document instance.

        Raises:
            DocumentNotFoundError: If the specified file does not exist.
            IngestionError: If the source is a directory or file read fails.
            UnsupportedFileTypeError: If the file extension is not supported.
            EmptyDocumentError: If allow_empty is False and the file is empty/whitespace.
        """
        path = Path(source).resolve()

        if not path.exists():
            raise DocumentNotFoundError(f"Document file not found at path: {path}")

        if not path.is_file():
            raise IngestionError(f"Specified source path is not a file: {path}")

        ext = path.suffix.lower()
        if ext not in self.supported_extensions:
            expected = ", ".join(sorted(self.supported_extensions))
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{ext}' for {self.__class__.__name__}. "
                f"Supported extensions: {expected}"
            )

        try:
            raw_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw_content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                raise IngestionError(f"Failed to read file at '{path}': {exc}") from exc
        except OSError as exc:
            raise IngestionError(f"Operating system error reading '{path}': {exc}") from exc

        if not allow_empty and not raw_content.strip():
            raise EmptyDocumentError(
                f"File '{path.name}' is empty or contains only whitespace characters."
            )

        content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()

        custom_id: UUID | None = kwargs.get("document_id")
        doc_id = custom_id if custom_id is not None else generate_document_id(content_hash)

        custom_title: str | None = kwargs.get("title")
        title = custom_title if custom_title else self._extract_title(path, raw_content)

        file_metadata: dict[str, Any] = {
            "source": str(path),
            "file_name": path.name,
            "file_stem": path.stem,
            "file_extension": ext,
            "file_size": path.stat().st_size,
        }

        extra_metadata: dict[str, Any] = kwargs.get("metadata", {})
        merged_metadata = {**file_metadata, **extra_metadata}

        return Document(
            id=doc_id,
            title=title,
            source_type=self.mime_type,
            content_hash=content_hash,
            raw_content=raw_content,
            file_path=str(path),
            metadata=merged_metadata,
            status=DocumentStatus.PENDING,
        )
