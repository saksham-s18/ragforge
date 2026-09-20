from abc import ABC, abstractmethod
from collections.abc import Sequence


class BaseEmbeddingProvider(ABC):
    """Abstract port for generating dense vector embeddings."""

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a collection of text strings."""

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
