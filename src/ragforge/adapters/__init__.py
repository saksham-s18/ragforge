"""Concrete infrastructure and data-source adapters implementing domain ports."""

from ragforge.adapters.chunkers import (
    DeterministicChunker,
    approximate_token_count,
)
from ragforge.adapters.embeddings import (
    DeterministicEmbeddingProvider,
    FastEmbedProvider,
)
from ragforge.adapters.loaders import (
    BaseFileLoader,
    MarkdownDocumentLoader,
    TextDocumentLoader,
    generate_document_id,
)
from ragforge.adapters.state import (
    InMemoryIndexStateStore,
    JsonFileIndexStateStore,
)
from ragforge.adapters.vector_stores import (
    InMemoryVectorStore,
    QdrantVectorStore,
)

__all__ = [
    "BaseFileLoader",
    "DeterministicChunker",
    "DeterministicEmbeddingProvider",
    "FastEmbedProvider",
    "InMemoryIndexStateStore",
    "InMemoryVectorStore",
    "JsonFileIndexStateStore",
    "MarkdownDocumentLoader",
    "QdrantVectorStore",
    "TextDocumentLoader",
    "approximate_token_count",
    "generate_document_id",
]
