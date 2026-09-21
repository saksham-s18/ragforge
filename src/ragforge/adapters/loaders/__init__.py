"""Concrete document loader adapters for ingesting local files."""

from ragforge.adapters.loaders.base import (
    BaseFileLoader,
    generate_document_id,
)
from ragforge.adapters.loaders.markdown import MarkdownDocumentLoader
from ragforge.adapters.loaders.text import TextDocumentLoader

__all__ = [
    "BaseFileLoader",
    "MarkdownDocumentLoader",
    "TextDocumentLoader",
    "generate_document_id",
]
