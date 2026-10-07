from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragforge.domain.models import RetrievedChunk


class BaseScoreFusion(ABC):
    """Abstract port for fusing and re-ranking multi-retriever candidate result lists."""

    @abstractmethod
    def fuse(
        self,
        dense_results: Sequence[RetrievedChunk],
        lexical_results: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse and rank candidate results from dense and lexical retrieval runs.

        Args:
            dense_results: Ranked chunks retrieved via dense vector similarity.
            lexical_results: Ranked chunks retrieved via lexical BM25 search.
            top_k: Optional maximum number of fused results to return. If None,
                all fused candidates are returned.

        Returns:
            Ranked list of fused RetrievedChunk models ordered by descending fusion score.
        """
