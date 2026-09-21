from ragforge.adapters.loaders.base import BaseFileLoader
from ragforge.domain.enums import MimeType


class TextDocumentLoader(BaseFileLoader):
    """Loader adapter for plain text (.txt, .text) documents."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".txt", ".text"}

    @property
    def mime_type(self) -> MimeType:
        return MimeType.TXT
