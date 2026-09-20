from abc import ABC, abstractmethod
from typing import Any

from ragforge.domain.models import Document


class BaseDocumentLoader(ABC):
    """Abstract port for parsing documents from raw bytes or files into domain Documents."""

    @abstractmethod
    async def load(self, source: Any, **kwargs: Any) -> Document:
        """Parse source content into a domain Document."""
