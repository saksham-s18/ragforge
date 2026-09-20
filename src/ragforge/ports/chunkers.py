from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragforge.domain.models import Chunk, Document


class BaseChunker(ABC):
    """Abstract port for segmenting a Document into discrete Chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Split a Document into chunks with metadata."""
