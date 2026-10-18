import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from ragforge.domain.enums import SearchStrategy
from ragforge.domain.exceptions import VectorStoreError
from ragforge.domain.models import Chunk, Query, RetrievedChunk
from ragforge.ports.embeddings import BaseEmbeddingProvider
from ragforge.ports.fusion import BaseScoreFusion
from ragforge.ports.lexical import BaseLexicalIndex
from ragforge.ports.vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    """Application retrieval service orchestrating dense, sparse, and hybrid search.

    Coordinates dense retrieval by translating raw text or domain Query objects into
    vector representations via an injected BaseEmbeddingProvider port and delegating
    k-NN search to an injected BaseVectorStore port. Optionally coordinates lexical search
    via an injected BaseLexicalIndex port, and multi-channel result combination via an
    injected BaseScoreFusion port.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        lexical_index: BaseLexicalIndex | None = None,
        fusion: BaseScoreFusion | None = None,
        default_search_strategy: SearchStrategy = SearchStrategy.DENSE,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._fusion = fusion
        self._default_search_strategy = default_search_strategy

    async def retrieve(
        self,
        query: Query | str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        search_strategy: SearchStrategy | None = None,
    ) -> list[RetrievedChunk]:
        """Execute retrieval for a query string or domain Query model.

        Supports dense vector search, lexical BM25 search, or hybrid multi-channel
        search with score fusion.

        Args:
            query: Query string or domain Query object.
            top_k: Optional top_k override. If None, falls back to Query.top_k or default 10.
            filters: Optional metadata filters override.
            score_threshold: Optional minimum cosine similarity score threshold (dense only).
            search_strategy: Optional search strategy override (DENSE, SPARSE, HYBRID).

        Returns:
            List of RetrievedChunk models sorted by descending score with ranks.

        Raises:
            VectorStoreError: If top_k <= 0 or if both retrieval channels fail.
            ValueError: If a required port for the requested search strategy is missing.
        """
        if isinstance(query, str):
            raw_query = query
            effective_top_k = top_k if top_k is not None else 10
            effective_filters = filters
            effective_threshold = score_threshold
            effective_strategy = search_strategy or self._default_search_strategy
        else:
            raw_query = query.raw_query
            effective_top_k = top_k if top_k is not None else query.top_k
            effective_filters = filters if filters is not None else query.filters
            effective_threshold = (
                score_threshold if score_threshold is not None else query.score_threshold
            )
            if search_strategy is not None:
                effective_strategy = search_strategy
            elif "search_strategy" in query.model_fields_set:
                effective_strategy = query.search_strategy
            else:
                effective_strategy = self._default_search_strategy

        if effective_top_k <= 0:
            raise VectorStoreError(f"top_k must be a positive integer, got {effective_top_k}.")

        if not raw_query or not raw_query.strip():
            return []

        if effective_strategy == SearchStrategy.DENSE:
            return await self._dense_search(
                raw_query=raw_query,
                top_k=effective_top_k,
                filters=effective_filters,
                score_threshold=effective_threshold,
            )

        if effective_strategy == SearchStrategy.SPARSE:
            return await self._sparse_search(
                raw_query=raw_query,
                top_k=effective_top_k,
                filters=effective_filters,
            )

        if effective_strategy == SearchStrategy.HYBRID:
            return await self._hybrid_search(
                raw_query=raw_query,
                top_k=effective_top_k,
                filters=effective_filters,
                score_threshold=effective_threshold,
            )

        raise ValueError(f"Unsupported search strategy: {effective_strategy}")

    async def _dense_search(
        self,
        raw_query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = await self._embedding_provider.embed_query(raw_query)
        retrieved_chunks = await self._vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        if score_threshold is not None:
            retrieved_chunks = [c for c in retrieved_chunks if c.score >= score_threshold]

        results: list[RetrievedChunk] = []
        for rank, item in enumerate(retrieved_chunks, start=1):
            results.append(
                RetrievedChunk(
                    chunk=item.chunk,
                    score=item.score,
                    retrieval_type="dense",
                    rank=rank,
                    dense_score=item.dense_score if item.dense_score is not None else item.score,
                    lexical_score=None,
                )
            )
        return results

    async def _sparse_search(
        self,
        raw_query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._lexical_index is None:
            raise ValueError("Lexical index is required for SPARSE retrieval strategy.")

        retrieved_chunks = await self._lexical_index.search(
            query=raw_query,
            top_k=top_k,
            filters=filters,
        )

        results: list[RetrievedChunk] = []
        for rank, item in enumerate(retrieved_chunks, start=1):
            results.append(
                RetrievedChunk(
                    chunk=item.chunk,
                    score=item.score,
                    retrieval_type=item.retrieval_type or "lexical",
                    rank=rank,
                    dense_score=None,
                    lexical_score=(
                        item.lexical_score if item.lexical_score is not None else item.score
                    ),
                )
            )
        return results

    async def _hybrid_search(
        self,
        raw_query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        if self._lexical_index is None:
            raise ValueError("Lexical index is required for HYBRID retrieval strategy.")
        if self._fusion is None:
            raise ValueError("Score fusion is required for HYBRID retrieval strategy.")

        dense_task = self._dense_search(
            raw_query=raw_query,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )
        lexical_task = self._sparse_search(
            raw_query=raw_query,
            top_k=top_k,
            filters=filters,
        )

        dense_outcome, lexical_outcome = await asyncio.gather(
            dense_task,
            lexical_task,
            return_exceptions=True,
        )

        if isinstance(dense_outcome, BaseException):
            if isinstance(lexical_outcome, BaseException):
                logger.error(
                    "Hybrid retrieval: both channels failed. Dense error: %s; Lexical error: %s",
                    dense_outcome,
                    lexical_outcome,
                )
                raise VectorStoreError(
                    f"Both retrieval channels failed during hybrid retrieval: "
                    f"dense error: {dense_outcome}; lexical error: {lexical_outcome}"
                ) from dense_outcome

            logger.warning(
                "Dense retrieval failed during hybrid search; continuing with lexical: %s",
                dense_outcome,
            )
            return lexical_outcome

        if isinstance(lexical_outcome, BaseException):
            logger.warning(
                "Lexical retrieval failed during hybrid search; continuing with dense: %s",
                lexical_outcome,
            )
            return dense_outcome

        dense_results: list[RetrievedChunk] = dense_outcome
        lexical_results: list[RetrievedChunk] = lexical_outcome

        if not dense_results and not lexical_results:
            return []

        if not lexical_results:
            logger.debug("Lexical results empty in hybrid search; returning dense results.")
            return dense_results

        if not dense_results:
            logger.debug("Dense results empty in hybrid search; returning lexical results.")
            return lexical_results

        return self._fusion.fuse(
            dense_results=dense_results,
            lexical_results=lexical_results,
            top_k=top_k,
        )

    async def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Embed a collection of chunks and upsert them into the vector store and lexical index."""
        if not chunks:
            return
        texts = [chunk.content for chunk in chunks]
        vectors = await self._embedding_provider.embed_texts(texts)
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.dense_vector = vector
        await self._vector_store.upsert(chunks)
        if self._lexical_index is not None:
            await self._lexical_index.index_chunks(chunks)
