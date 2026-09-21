from collections.abc import Sequence
from typing import Any
from uuid import UUID

from ragforge.core.similarity import cosine_similarity
from ragforge.domain.exceptions import (
    VectorDimensionMismatchError,
    VectorStoreError,
)
from ragforge.domain.models import Chunk, RetrievedChunk
from ragforge.ports.vector_store import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """Thread-safe in-memory vector store adapter for local development and testing.

    Stores domain Chunk objects and dense vectors in a Python dictionary. Executes exact
    k-nearest neighbors (kNN) search using cosine similarity without requiring external
    vector database processes or persistence engines.
    """

    def __init__(self, dimension: int | None = None) -> None:
        if dimension is not None and dimension <= 0:
            raise ValueError(f"Vector store dimension must be greater than 0, got {dimension}")
        self.dimension = dimension
        self._storage: dict[UUID, Chunk] = {}

    def count(self) -> int:
        """Return the current number of stored chunks."""
        return len(self._storage)

    def __len__(self) -> int:
        return len(self._storage)

    async def get(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve an indexed chunk by its unique chunk identifier."""
        return self._storage.get(chunk_id)

    async def clear(self) -> None:
        """Remove all chunks from the in-memory store."""
        self._storage.clear()

    async def delete_by_chunk_id(self, chunk_id: UUID) -> bool:
        """Delete a single chunk by its ID. Returns True if deleted, False if not found."""
        if chunk_id in self._storage:
            del self._storage[chunk_id]
            return True
        return False

    async def delete_by_document_id(self, document_id: UUID) -> None:
        """Delete all chunks associated with the specified document ID."""
        to_delete = [
            cid for cid, chunk in self._storage.items() if chunk.document_id == document_id
        ]
        for cid in to_delete:
            del self._storage[cid]

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Store or update chunks along with their dense vector embeddings.

        Args:
            chunks: Collection of domain Chunk instances with populated dense_vector fields.

        Raises:
            VectorStoreError: If a chunk lacks a dense_vector embedding.
            VectorDimensionMismatchError: If a chunk vector dimension does not match
                store dimension.
        """
        for chunk in chunks:
            if chunk.dense_vector is None:
                raise VectorStoreError(f"Cannot upsert Chunk {chunk.id}: 'dense_vector' is None.")

            if self.dimension is not None and len(chunk.dense_vector) != self.dimension:
                raise VectorDimensionMismatchError(
                    f"Cannot upsert Chunk {chunk.id}: vector dimension {len(chunk.dense_vector)} "
                    f"does not match store configured dimension {self.dimension}."
                )

            # Insert or update in storage
            self._storage[chunk.id] = chunk

    def _matches_filters(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        """Evaluate whether a chunk satisfies the provided filter dictionary."""
        for key, expected_val in filters.items():
            if key == "document_id":
                if str(chunk.document_id) != str(expected_val):
                    return False
            elif key in chunk.metadata.extra:
                if chunk.metadata.extra[key] != expected_val:
                    return False
            elif hasattr(chunk.metadata, key):
                if getattr(chunk.metadata, key) != expected_val:
                    return False
            elif hasattr(chunk, key):
                if getattr(chunk, key) != expected_val:
                    return False
            else:
                return False
        return True

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform exact cosine similarity search against stored vectors.

        Args:
            query_vector: Dense query vector.
            top_k: Maximum number of nearest neighbors to return. Must be > 0.
            filters: Optional key-value criteria to filter candidate chunks.

        Returns:
            List of RetrievedChunk domain models ordered by descending cosine similarity.

        Raises:
            VectorStoreError: If query_vector is empty or top_k <= 0.
            VectorDimensionMismatchError: If query_vector length mismatches configured dimension.
        """
        if not query_vector:
            raise VectorStoreError("Query vector cannot be empty.")

        if self.dimension is not None and len(query_vector) != self.dimension:
            raise VectorDimensionMismatchError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"store dimension {self.dimension}."
            )

        if top_k <= 0:
            raise VectorStoreError(f"top_k must be a positive integer, got {top_k}.")

        if not self._storage:
            return []

        scored_candidates: list[tuple[float, Chunk]] = []

        for chunk in self._storage.values():
            if filters and not self._matches_filters(chunk, filters):
                continue

            if chunk.dense_vector is None:
                continue

            score = cosine_similarity(query_vector, chunk.dense_vector)
            scored_candidates.append((score, chunk))

        # Sort descending by similarity score
        scored_candidates.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(scored_candidates[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    retrieval_type="dense",
                    rank=rank,
                )
            )

        return results
