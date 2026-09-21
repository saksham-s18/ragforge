from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, ScoredPoint, VectorParams

from ragforge.adapters.vector_stores.qdrant import QdrantVectorStore
from ragforge.domain.exceptions import (
    VectorDimensionMismatchError,
    VectorStoreConnectionError,
    VectorStoreError,
)
from ragforge.domain.models import Chunk, ChunkMetadata


def _create_chunk(
    dim: int = 4,
    document_id: UUID | None = None,
    content: str = "Test chunk content",
) -> Chunk:
    """Helper to instantiate Chunk models."""
    return Chunk(
        id=uuid4(),
        document_id=document_id or uuid4(),
        chunk_index=0,
        content=content,
        token_count=10,
        content_hash="testhash123",
        metadata=ChunkMetadata(
            section_header="Architecture",
            start_char_idx=0,
            end_char_idx=len(content),
            extra={"source": "unit_test"},
        ),
        dense_vector=[0.1] * dim,
    )


def test_qdrant_init_invalid_dimension() -> None:
    """Verify invalid vector dimensions raise ValueError."""
    with pytest.raises(ValueError, match="dimension must be greater than 0"):
        QdrantVectorStore(dimension=0)

    with pytest.raises(ValueError, match="dimension must be greater than 0"):
        QdrantVectorStore(dimension=-4)


@pytest.mark.asyncio
async def test_qdrant_init_collection_creates_if_not_exists() -> None:
    """Verify collection is created if it does not already exist."""
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.collection_exists.return_value = False

    store = QdrantVectorStore(
        client=mock_client,
        collection_name="test_col",
        dimension=64,
    )
    await store.init_collection()

    mock_client.collection_exists.assert_awaited_once_with("test_col")
    mock_client.create_collection.assert_awaited_once_with(
        collection_name="test_col",
        vectors_config=VectorParams(size=64, distance=Distance.COSINE),
    )


@pytest.mark.asyncio
async def test_qdrant_init_collection_validates_dimension() -> None:
    """Verify dimension mismatch against existing collection raises VectorDimensionMismatchError."""
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.collection_exists.return_value = True

    mock_info = MagicMock()
    mock_info.config.params.vectors = VectorParams(size=128, distance=Distance.COSINE)
    mock_client.get_collection.return_value = mock_info

    store = QdrantVectorStore(
        client=mock_client,
        collection_name="existing_col",
        dimension=64,  # Configured 64, existing is 128
    )

    with pytest.raises(VectorDimensionMismatchError, match="has vector dimension 128"):
        await store.init_collection()


@pytest.mark.asyncio
async def test_qdrant_upsert_validation() -> None:
    """Verify upsert enforces presence of dense_vector and dimension matching."""
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.collection_exists.return_value = True
    mock_info = MagicMock()
    mock_info.config.params.vectors = VectorParams(size=4, distance=Distance.COSINE)
    mock_client.get_collection.return_value = mock_info

    store = QdrantVectorStore(client=mock_client, dimension=4)

    # Missing dense vector
    chunk_no_vec = _create_chunk(dim=4)
    chunk_no_vec.dense_vector = None
    with pytest.raises(VectorStoreError, match="'dense_vector' is None"):
        await store.upsert([chunk_no_vec])

    # Dimension mismatch
    chunk_wrong_dim = _create_chunk(dim=8)
    with pytest.raises(VectorDimensionMismatchError, match="vector dimension 8"):
        await store.upsert([chunk_wrong_dim])


@pytest.mark.asyncio
async def test_qdrant_upsert_payload_and_points() -> None:
    """Verify points and payloads are mapped correctly to Qdrant PointStructs."""
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.collection_exists.return_value = True
    mock_info = MagicMock()
    mock_info.config.params.vectors = VectorParams(size=4, distance=Distance.COSINE)
    mock_client.get_collection.return_value = mock_info

    store = QdrantVectorStore(client=mock_client, dimension=4)
    chunk = _create_chunk(dim=4)

    await store.upsert([chunk])

    mock_client.upsert.assert_awaited_once()
    call_kwargs = mock_client.upsert.await_args.kwargs
    assert call_kwargs["collection_name"] == "ragforge_chunks"
    points = call_kwargs["points"]
    assert len(points) == 1
    pt = points[0]
    assert pt.id == str(chunk.id)
    assert pt.vector == chunk.dense_vector
    assert pt.payload["document_id"] == str(chunk.document_id)
    assert pt.payload["section_header"] == "Architecture"
    assert pt.payload["extra"] == {"source": "unit_test"}


@pytest.mark.asyncio
async def test_qdrant_search_validation_and_results() -> None:
    """Verify search input validation and conversion of Qdrant points to RetrievedChunk."""
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.collection_exists.return_value = True
    mock_info = MagicMock()
    mock_info.config.params.vectors = VectorParams(size=4, distance=Distance.COSINE)
    mock_client.get_collection.return_value = mock_info

    store = QdrantVectorStore(client=mock_client, dimension=4)

    # Empty query
    with pytest.raises(VectorStoreError, match="Query vector cannot be empty"):
        await store.search(query_vector=[], top_k=5)

    # Invalid top-k
    with pytest.raises(VectorStoreError, match="top_k must be a positive integer"):
        await store.search(query_vector=[0.1, 0.2, 0.3, 0.4], top_k=0)

    # Dimension mismatch
    with pytest.raises(VectorDimensionMismatchError, match="Query vector dimension 2"):
        await store.search(query_vector=[0.1, 0.2], top_k=5)

    # Successful search
    cid = uuid4()
    did = uuid4()
    mock_response = MagicMock()
    mock_response.points = [
        ScoredPoint(
            id=str(cid),
            version=0,
            score=0.95,
            payload={
                "chunk_id": str(cid),
                "document_id": str(did),
                "chunk_index": 0,
                "content": "Retrieved content",
                "content_hash": "hash123",
                "token_count": 5,
                "section_header": "Overview",
                "start_char_idx": 0,
                "end_char_idx": 17,
                "extra": {"tier": "primary"},
            },
            vector=[0.1, 0.2, 0.3, 0.4],
        )
    ]
    mock_client.query_points.return_value = mock_response

    results = await store.search(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        top_k=1,
        filters={"tier": "primary"},
    )

    assert len(results) == 1
    r = results[0]
    assert r.chunk.id == cid
    assert r.chunk.document_id == did
    assert r.chunk.content == "Retrieved content"
    assert r.chunk.metadata.section_header == "Overview"
    assert r.chunk.metadata.extra == {"tier": "primary"}
    assert r.score == 0.95
    assert r.rank == 1
    assert r.retrieval_type == "dense"


@pytest.mark.asyncio
async def test_qdrant_connection_error_translation() -> None:
    """Verify low-level connection errors are caught and re-raised as VectorStoreConnectionError."""
    mock_client = AsyncMock(spec=AsyncQdrantClient)
    mock_client.collection_exists.side_effect = ResponseHandlingException(
        Exception("Connection refused")
    )

    store = QdrantVectorStore(client=mock_client, dimension=4)

    with pytest.raises(VectorStoreConnectionError, match="Failed to connect to Qdrant"):
        await store.init_collection()


@pytest.mark.asyncio
async def test_qdrant_in_memory_full_lifecycle() -> None:
    """Verify complete QdrantVectorStore lifecycle using Qdrant's in-memory engine."""
    client = AsyncQdrantClient(location=":memory:")
    store = QdrantVectorStore(
        client=client,
        collection_name="lifecycle_col",
        dimension=4,
    )

    await store.init_collection()

    doc_id = uuid4()
    c1 = _create_chunk(dim=4, document_id=doc_id, content="First chunk")
    c2 = _create_chunk(dim=4, document_id=doc_id, content="Second chunk")

    # Upsert
    await store.upsert([c1, c2])
    assert await store.count() == 2

    # Get
    retrieved = await store.get(c1.id)
    assert retrieved is not None
    assert retrieved.id == c1.id
    assert retrieved.content == "First chunk"

    # Search
    search_results = await store.search(query_vector=[0.1, 0.1, 0.1, 0.1], top_k=2)
    assert len(search_results) == 2
    assert search_results[0].rank == 1
    assert search_results[1].rank == 2

    # Delete by chunk ID
    deleted = await store.delete_by_chunk_id(c1.id)
    assert deleted is True
    assert await store.count() == 1
    assert await store.get(c1.id) is None

    # Clear
    await store.clear()
    assert await store.count() == 0
