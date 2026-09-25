"""Abstract interfaces (ports) defining system component contracts."""

from ragforge.ports.chunkers import BaseChunker
from ragforge.ports.embeddings import BaseEmbeddingProvider
from ragforge.ports.loaders import BaseDocumentLoader
from ragforge.ports.state import BaseIndexStateStore
from ragforge.ports.vector_store import BaseVectorStore

__all__ = [
    "BaseChunker",
    "BaseDocumentLoader",
    "BaseEmbeddingProvider",
    "BaseIndexStateStore",
    "BaseVectorStore",
]
