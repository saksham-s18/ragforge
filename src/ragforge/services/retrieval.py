from collections.abc import Sequence
from typing import Any

from ragforge.domain.exceptions import VectorStoreError
from ragforge.domain.models import Chunk, Query, RetrievedChunk
from ragforge.ports.embeddings import BaseEmbeddingProvider
from ragforge.ports.vector_store import BaseVectorStore


class RetrievalService:
    """Application retrieval service orchestrating query embedding and vector search.

    Coordinates dense retrieval by translating raw text or domain Query objects into
    vector representations via an injected BaseEmbeddingProvider port, and delegating
    k-NN search to an injected BaseVectorStore port.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def retrieve(
        self,
        query: Query | str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Execute dense vector retrieval for a query string or domain Query model.

        Args:
            query: Query string or domain Query object.
            top_k: Optional top_k override. If None, falls back to Query.top_k or default 10.
            filters: Optional metadata filters override.
            score_threshold: Optional minimum cosine similarity score threshold.

        Returns:
            List of RetrievedChunk models sorted by descending similarity with ranks.

        Raises:
            VectorStoreError: If top_k <= 0.
        """
        if isinstance(query, str):
            raw_query = query
            effective_top_k = top_k if top_k is not None else 10
            effective_filters = filters
            effective_threshold = score_threshold
        else:
            raw_query = query.raw_query
            effective_top_k = top_k if top_k is not None else query.top_k
            effective_filters = filters if filters is not None else query.filters
            effective_threshold = (
                score_threshold if score_threshold is not None else query.score_threshold
            )

        if effective_top_k <= 0:
            raise VectorStoreError(f"top_k must be a positive integer, got {effective_top_k}.")

        if not raw_query or not raw_query.strip():
            return []

        query_vector = await self._embedding_provider.embed_query(raw_query)
        retrieved_chunks = await self._vector_store.search(
            query_vector=query_vector,
            top_k=effective_top_k,
            filters=effective_filters,
        )

        if effective_threshold is not None:
            filtered = [c for c in retrieved_chunks if c.score >= effective_threshold]
            retrieved_chunks = [
                RetrievedChunk(
                    chunk=item.chunk,
                    score=item.score,
                    retrieval_type=item.retrieval_type,
                    rank=new_rank,
                )
                for new_rank, item in enumerate(filtered, start=1)
            ]

        return retrieved_chunks

    async def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Embed a collection of chunks and upsert them into the vector store."""
        if not chunks:
            return
        texts = [chunk.content for chunk in chunks]
        vectors = await self._embedding_provider.embed_texts(texts)
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.dense_vector = vector
        await self._vector_store.upsert(chunks)
