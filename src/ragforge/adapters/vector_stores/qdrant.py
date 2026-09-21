from collections.abc import Sequence
from typing import Any
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ragforge.domain.exceptions import (
    VectorDimensionMismatchError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from ragforge.domain.models import Chunk, ChunkMetadata, RetrievedChunk
from ragforge.ports.vector_store import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """Production vector-store adapter interfacing with a Qdrant cluster or standalone instance.

    Stores Chunk embeddings as dense vectors indexed via Hierarchical Navigable Small World (HNSW)
    graphs, and retains full chunk and document provenance within the point payload.
    """

    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "ragforge_chunks",
        dimension: int = 64,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"Vector store dimension must be greater than 0, got {dimension}")

        self.collection_name = collection_name
        self.dimension = dimension
        self._url = url or "http://localhost:6333"
        self._api_key = api_key

        if client is not None:
            self._client = client
        else:
            self._client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key,
                check_compatibility=False,
            )

        self._initialized = False

    async def init_collection(self) -> None:
        """Create the collection if it does not exist, and validate vector dimensionality."""
        try:
            exists = await self._client.collection_exists(self.collection_name)
            if not exists:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                self._initialized = True
                return

            info = await self._client.get_collection(self.collection_name)
            vectors_config = info.config.params.vectors
            existing_size: int | None = None
            if isinstance(vectors_config, VectorParams):
                existing_size = vectors_config.size
            elif hasattr(vectors_config, "size"):
                existing_size = vectors_config.size  # type: ignore[union-attr]

            if existing_size is not None and existing_size != self.dimension:
                raise VectorDimensionMismatchError(
                    f"Qdrant collection '{self.collection_name}' has vector dimension "
                    f"{existing_size}, which does not match configured dimension {self.dimension}."
                )
            self._initialized = True
        except VectorDimensionMismatchError:
            raise
        except (ResponseHandlingException, ConnectionError, OSError) as exc:
            raise VectorStoreConnectionError(
                f"Failed to connect to Qdrant at {self._url}: {exc}"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize Qdrant collection '{self.collection_name}': {exc}"
            ) from exc

    async def _ensure_initialized(self) -> None:
        """Lazy initialization hook to guarantee collection is ready before operations."""
        if not self._initialized:
            await self.init_collection()

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Store or update chunks along with their vector representations and provenance."""
        await self._ensure_initialized()
        if not chunks:
            return

        points: list[PointStruct] = []
        for chunk in chunks:
            if chunk.dense_vector is None:
                raise VectorStoreError(f"Cannot upsert Chunk {chunk.id}: 'dense_vector' is None.")

            if len(chunk.dense_vector) != self.dimension:
                raise VectorDimensionMismatchError(
                    f"Cannot upsert Chunk {chunk.id}: vector dimension {len(chunk.dense_vector)} "
                    f"does not match configured dimension {self.dimension}."
                )

            payload = {
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "content_hash": chunk.content_hash,
                "token_count": chunk.token_count,
                "page_number": chunk.metadata.page_number,
                "section_header": chunk.metadata.section_header,
                "start_char_idx": chunk.metadata.start_char_idx,
                "end_char_idx": chunk.metadata.end_char_idx,
                "extra": chunk.metadata.extra,
            }

            points.append(
                PointStruct(
                    id=str(chunk.id),
                    vector=chunk.dense_vector,
                    payload=payload,
                )
            )

        try:
            await self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except (ResponseHandlingException, ConnectionError, OSError) as exc:
            raise VectorStoreConnectionError(
                f"Connection error while upserting points to Qdrant: {exc}"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert points to Qdrant: {exc}") from exc

    def _build_filter(self, filters: dict[str, Any]) -> Filter:
        """Translate key-value filter dictionary to Qdrant Filter conditions."""
        conditions: list[Condition] = []
        for key, val in filters.items():
            if key in (
                "document_id",
                "chunk_id",
                "chunk_index",
                "content_hash",
                "token_count",
                "section_header",
                "page_number",
                "start_char_idx",
                "end_char_idx",
            ):
                field_key = key
            elif key.startswith("extra."):
                field_key = key
            else:
                field_key = f"extra.{key}"

            match_val = str(val) if isinstance(val, UUID) else val
            conditions.append(
                FieldCondition(
                    key=field_key,
                    match=MatchValue(value=match_val),
                )
            )
        return Filter(must=conditions)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k nearest chunks matching the query vector and optional filters."""
        await self._ensure_initialized()

        if not query_vector:
            raise VectorStoreError("Query vector cannot be empty.")

        if len(query_vector) != self.dimension:
            raise VectorDimensionMismatchError(
                f"Query vector dimension {len(query_vector)} does not match "
                f"store dimension {self.dimension}."
            )

        if top_k <= 0:
            raise VectorStoreError(f"top_k must be a positive integer, got {top_k}.")

        qdrant_filter: Filter | None = None
        if filters:
            qdrant_filter = self._build_filter(filters)

        try:
            response = await self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=True,
            )
        except (ResponseHandlingException, ConnectionError, OSError) as exc:
            raise VectorStoreConnectionError(
                f"Connection error while searching Qdrant collection: {exc}"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to execute search on Qdrant: {exc}") from exc

        results: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            vector_data: list[float] | None = None
            if point.vector is not None and isinstance(point.vector, list):
                vector_data = [float(x) for x in point.vector if isinstance(x, (int, float))]

            chunk = Chunk(
                id=UUID(str(payload.get("chunk_id", point.id))),
                document_id=UUID(str(payload["document_id"])),
                chunk_index=int(payload.get("chunk_index", 0)),
                content=str(payload.get("content", "")),
                token_count=int(payload.get("token_count", 0)),
                content_hash=str(payload.get("content_hash", "")),
                metadata=ChunkMetadata(
                    page_number=payload.get("page_number"),
                    section_header=payload.get("section_header"),
                    start_char_idx=payload.get("start_char_idx"),
                    end_char_idx=payload.get("end_char_idx"),
                    extra=payload.get("extra", {}),
                ),
                dense_vector=vector_data,
            )

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=float(point.score),
                    retrieval_type="dense",
                    rank=rank,
                )
            )

        return results

    async def get(self, chunk_id: UUID) -> Chunk | None:
        """Retrieve a single Chunk by its UUID."""
        await self._ensure_initialized()
        try:
            records = await self._client.retrieve(
                collection_name=self.collection_name,
                ids=[str(chunk_id)],
                with_payload=True,
                with_vectors=True,
            )
        except (ResponseHandlingException, ConnectionError, OSError) as exc:
            raise VectorStoreConnectionError(
                f"Connection error retrieving point from Qdrant: {exc}"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to retrieve point from Qdrant: {exc}") from exc

        if not records:
            return None

        record = records[0]
        payload = record.payload or {}
        vector_data: list[float] | None = None
        if record.vector is not None and isinstance(record.vector, list):
            vector_data = [float(x) for x in record.vector if isinstance(x, (int, float))]

        return Chunk(
            id=UUID(str(payload.get("chunk_id", record.id))),
            document_id=UUID(str(payload["document_id"])),
            chunk_index=int(payload.get("chunk_index", 0)),
            content=str(payload.get("content", "")),
            token_count=int(payload.get("token_count", 0)),
            content_hash=str(payload.get("content_hash", "")),
            metadata=ChunkMetadata(
                page_number=payload.get("page_number"),
                section_header=payload.get("section_header"),
                start_char_idx=payload.get("start_char_idx"),
                end_char_idx=payload.get("end_char_idx"),
                extra=payload.get("extra", {}),
            ),
            dense_vector=vector_data,
        )

    async def delete_by_chunk_id(self, chunk_id: UUID) -> bool:
        """Delete a single chunk by its ID. Returns True if deleted, False if not found."""
        await self._ensure_initialized()
        existing = await self.get(chunk_id)
        if existing is None:
            return False

        try:
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=[str(chunk_id)],
            )
            return True
        except (ResponseHandlingException, ConnectionError, OSError) as exc:
            raise VectorStoreConnectionError(
                f"Connection error deleting point from Qdrant: {exc}"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete point from Qdrant: {exc}") from exc

    async def delete_by_document_id(self, document_id: UUID) -> None:
        """Delete all chunks associated with a document ID."""
        await self._ensure_initialized()
        try:
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id)),
                        )
                    ]
                ),
            )
        except (ResponseHandlingException, ConnectionError, OSError) as exc:
            raise VectorStoreConnectionError(
                f"Connection error deleting by document_id in Qdrant: {exc}"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete by document_id in Qdrant: {exc}") from exc

    async def clear(self) -> None:
        """Clear all points from the collection without deleting the collection schema."""
        await self._ensure_initialized()
        try:
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(),
            )
        except (ResponseHandlingException, ConnectionError, OSError) as exc:
            raise VectorStoreConnectionError(
                f"Connection error clearing Qdrant collection: {exc}"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to clear Qdrant collection: {exc}") from exc

    async def count(self) -> int:
        """Return the count of points in the collection."""
        await self._ensure_initialized()
        try:
            res = await self._client.count(collection_name=self.collection_name)
            return res.count
        except (ResponseHandlingException, ConnectionError, OSError) as exc:
            raise VectorStoreConnectionError(
                f"Connection error counting points in Qdrant: {exc}"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(f"Failed to count points in Qdrant: {exc}") from exc
