from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragforge.domain.models import RetrievedChunk


class BaseReranker(ABC):
    """Abstract port for reranking candidate retrieved chunks against a query.

    Coordinates second-stage relevance scoring between initial vector candidate
    generation and final context assembly. Implementations must preserve original
    retrieval provenance and initial scores.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank candidate chunks according to query relevance.

        Args:
            query: User query string to score chunks against.
            chunks: Candidate RetrievedChunk instances from initial retrieval.
            top_k: Optional maximum number of top reranked chunks to return. If None,
                all reranked chunks are returned.

        Returns:
            List of reranked RetrievedChunks sorted by descending relevance with updated
            ranks (1 to N) and rerank scores preserved, while maintaining original
            vector retrieval scores and chunk provenance.

        Raises:
            ValueError: If top_k is non-None and <= 0.
            RerankingError: If reranking computation fails.
        """
        ...
