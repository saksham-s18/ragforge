from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from ragforge.adapters.vector_stores.qdrant import QdrantVectorStore
from ragforge.domain.exceptions import VectorDimensionMismatchError
from ragforge.domain.models import Chunk, ChunkMetadata

QDRANT_TEST_URL = "http://localhost:6333"


@pytest.fixture
async def qdrant_client() -> AsyncQdrantClient:
    """Fixture providing a connected AsyncQdrantClient."""
    client = AsyncQdrantClient(url=QDRANT_TEST_URL, check_compatibility=False)
    yield client
    await client.close()


@pytest.fixture
async def temp_collection(qdrant_client: AsyncQdrantClient) -> str:
    """Fixture ensuring a fresh ephemeral collection is deleted after test execution."""
    collection_name = f"test_col_{uuid4().hex[:8]}"
    yield collection_name
    try:
        if await qdrant_client.collection_exists(collection_name):
            await qdrant_client.delete_collection(collection_name)
    except Exception:
        pass


def _create_chunk(dim: int = 4, document_id=None, content: str = "Integration chunk") -> Chunk:
    """Helper factory for Chunk entities in integration tests."""
    return Chunk(
        id=uuid4(),
        document_id=document_id or uuid4(),
        chunk_index=0,
        content=content,
        token_count=8,
        content_hash="hash_integration",
        metadata=ChunkMetadata(
            section_header="Integration Header",
            start_char_idx=0,
            end_char_idx=len(content),
            extra={"env": "integration_test"},
        ),
        dense_vector=[0.25] * dim,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_integration_collection_lifecycle(
    qdrant_client: AsyncQdrantClient, temp_collection: str
) -> None:
    """Verify collection creation and idempotent re-initialization on real Qdrant."""
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=4,
    )

    assert not await qdrant_client.collection_exists(temp_collection)
    await store.init_collection()
    assert await qdrant_client.collection_exists(temp_collection)

    # Calling init_collection again should safely no-op without recreation
    await store.init_collection()
    assert await qdrant_client.collection_exists(temp_collection)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_integration_dimension_validation(
    qdrant_client: AsyncQdrantClient, temp_collection: str
) -> None:
    """Verify connecting to an existing collection with incompatible dimension raises error."""
    store_dim4 = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=4,
    )
    await store_dim4.init_collection()

    # Second store instance targeting same collection but configured with dimension 8
    store_dim8 = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=8,
    )
    with pytest.raises(VectorDimensionMismatchError, match="has vector dimension 4"):
        await store_dim8.init_collection()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_integration_upsert_and_search(
    qdrant_client: AsyncQdrantClient, temp_collection: str
) -> None:
    """Verify upserting points and executing top-k search with descending score ordering."""
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=2,
    )
    await store.init_collection()

    # c1 aligns with [1.0, 0.0], c2 is orthogonal [0.0, 1.0]
    c1 = _create_chunk(dim=2, content="Perfect match chunk")
    c1.dense_vector = [1.0, 0.0]
    c2 = _create_chunk(dim=2, content="Orthogonal chunk")
    c2.dense_vector = [0.0, 1.0]

    await store.upsert([c1, c2])
    assert await store.count() == 2

    # Search for [1.0, 0.0]
    results = await store.search(query_vector=[1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == c1.id
    assert results[0].score == pytest.approx(1.0, rel=1e-3)
    assert results[0].rank == 1

    assert results[1].chunk.id == c2.id
    assert results[1].rank == 2
    assert results[0].score >= results[1].score


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_integration_metadata_and_provenance(
    qdrant_client: AsyncQdrantClient, temp_collection: str
) -> None:
    """Verify chunk metadata, section headers, character spans, and extras are preserved."""
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=2,
    )
    await store.init_collection()

    doc_id = uuid4()
    chunk = _create_chunk(dim=2, document_id=doc_id, content="Provenance preservation check")
    chunk.dense_vector = [1.0, 0.0]
    chunk.metadata.section_header = "Section 4.1"
    chunk.metadata.start_char_idx = 100
    chunk.metadata.end_char_idx = 132
    chunk.metadata.extra = {"author": "Alex", "priority": "high"}

    await store.upsert([chunk])

    retrieved = await store.get(chunk.id)
    assert retrieved is not None
    assert retrieved.id == chunk.id
    assert retrieved.document_id == doc_id
    assert retrieved.metadata.section_header == "Section 4.1"
    assert retrieved.metadata.start_char_idx == 100
    assert retrieved.metadata.end_char_idx == 132
    assert retrieved.metadata.extra == {"author": "Alex", "priority": "high"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_integration_delete_by_chunk_id(
    qdrant_client: AsyncQdrantClient, temp_collection: str
) -> None:
    """Verify deleting a point by chunk ID removes it from real Qdrant storage."""
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=2,
    )
    await store.init_collection()

    c1 = _create_chunk(dim=2)
    c1.dense_vector = [1.0, 0.0]
    await store.upsert([c1])
    assert await store.count() == 1

    deleted = await store.delete_by_chunk_id(c1.id)
    assert deleted is True
    assert await store.count() == 0
    assert await store.get(c1.id) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_integration_delete_by_document_id(
    qdrant_client: AsyncQdrantClient, temp_collection: str
) -> None:
    """Verify deleting points matching a document ID removes only those associated points."""
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=2,
    )
    await store.init_collection()

    doc_a = uuid4()
    doc_b = uuid4()

    c1 = _create_chunk(dim=2, document_id=doc_a)
    c1.dense_vector = [1.0, 0.0]
    c2 = _create_chunk(dim=2, document_id=doc_a)
    c2.dense_vector = [0.5, 0.5]
    c3 = _create_chunk(dim=2, document_id=doc_b)
    c3.dense_vector = [0.0, 1.0]

    await store.upsert([c1, c2, c3])
    assert await store.count() == 3

    await store.delete_by_document_id(doc_a)
    assert await store.count() == 1
    assert await store.get(c1.id) is None
    assert await store.get(c2.id) is None
    assert await store.get(c3.id) is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_qdrant_integration_clear_and_empty_behavior(
    qdrant_client: AsyncQdrantClient, temp_collection: str
) -> None:
    """Verify clear empties collection and searching empty collection returns empty list."""
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=2,
    )
    await store.init_collection()

    c1 = _create_chunk(dim=2)
    c1.dense_vector = [1.0, 0.0]
    await store.upsert([c1])
    assert await store.count() == 1

    await store.clear()
    assert await store.count() == 0

    results = await store.search(query_vector=[1.0, 0.0], top_k=5)
    assert results == []
