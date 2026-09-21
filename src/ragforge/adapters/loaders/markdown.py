import re
from pathlib import Path

from ragforge.adapters.loaders.base import BaseFileLoader
from ragforge.domain.enums import MimeType

_HEADING_TITLE_REGEX = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


class MarkdownDocumentLoader(BaseFileLoader):
    """Loader adapter for Markdown (.md, .markdown) documents."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".md", ".markdown"}

    @property
    def mime_type(self) -> MimeType:
        return MimeType.MARKDOWN

    def _extract_title(self, path: Path, content: str) -> str:
        """Derive document title from first Markdown heading or fall back to filename."""
        match = _HEADING_TITLE_REGEX.search(content)
        if match:
            return match.group(1).strip()
        return path.name
