from uuid import uuid4

import pytest

from ragforge.adapters.vector_stores import InMemoryVectorStore
from ragforge.domain.exceptions import (
    VectorDimensionMismatchError,
    VectorStoreError,
)
from ragforge.domain.models import Chunk, ChunkMetadata


def _create_chunk(
    dense_vector: list[float] | None = None,
    document_id=None,
    content: str = "Test chunk content",
    extra: dict | None = None,
) -> Chunk:
    """Helper factory for Chunk entities."""
    return Chunk(
        id=uuid4(),
        document_id=document_id or uuid4(),
        chunk_index=0,
        content=content,
        token_count=10,
        content_hash="abc123hash",
        metadata=ChunkMetadata(
            section_header="Introduction",
            start_char_idx=0,
            end_char_idx=len(content),
            extra=extra or {"author": "Tester"},
        ),
        dense_vector=dense_vector,
    )


@pytest.mark.asyncio
async def test_vector_store_upsert_and_get() -> None:
    """Verify upserting and getting chunks from InMemoryVectorStore."""
    store = InMemoryVectorStore(dimension=3)
    chunk = _create_chunk(dense_vector=[1.0, 0.0, 0.0])

    await store.upsert([chunk])
    assert store.count() == 1
    assert len(store) == 1

    retrieved = await store.get(chunk.id)
    assert retrieved is not None
    assert retrieved.id == chunk.id
    assert retrieved.content == chunk.content
    assert retrieved.dense_vector == [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_vector_store_upsert_overwrite() -> None:
    """Verify upserting an existing chunk ID updates its content and vector."""
    store = InMemoryVectorStore(dimension=2)
    chunk = _create_chunk(dense_vector=[1.0, 0.0], content="Initial version")

    await store.upsert([chunk])
    assert (await store.get(chunk.id)).content == "Initial version"  # type: ignore[union-attr]

    updated_chunk = Chunk(
        id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=0,
        content="Updated version",
        token_count=10,
        content_hash="newhash",
        dense_vector=[0.0, 1.0],
    )
    await store.upsert([updated_chunk])

    assert store.count() == 1
    retrieved = await store.get(chunk.id)
    assert retrieved is not None
    assert retrieved.content == "Updated version"
    assert retrieved.dense_vector == [0.0, 1.0]


@pytest.mark.asyncio
async def test_vector_store_delete_by_chunk_id() -> None:
    """Verify deleting a chunk by its specific chunk ID."""
    store = InMemoryVectorStore(dimension=2)
    c1 = _create_chunk(dense_vector=[1.0, 0.0])
    c2 = _create_chunk(dense_vector=[0.0, 1.0])

    await store.upsert([c1, c2])
    assert store.count() == 2

    deleted = await store.delete_by_chunk_id(c1.id)
    assert deleted is True
    assert store.count() == 1
    assert await store.get(c1.id) is None
    assert await store.get(c2.id) is not None

    deleted_again = await store.delete_by_chunk_id(c1.id)
    assert deleted_again is False


@pytest.mark.asyncio
async def test_vector_store_delete_by_document_id() -> None:
    """Verify deleting all chunks associated with a parent document ID."""
    store = InMemoryVectorStore(dimension=2)
    doc_id_1 = uuid4()
    doc_id_2 = uuid4()

    c1 = _create_chunk(dense_vector=[1.0, 0.0], document_id=doc_id_1)
    c2 = _create_chunk(dense_vector=[0.5, 0.5], document_id=doc_id_1)
    c3 = _create_chunk(dense_vector=[0.0, 1.0], document_id=doc_id_2)

    await store.upsert([c1, c2, c3])
    assert store.count() == 3

    await store.delete_by_document_id(doc_id_1)
    assert store.count() == 1
    assert await store.get(c1.id) is None
    assert await store.get(c2.id) is None
    assert await store.get(c3.id) is not None


@pytest.mark.asyncio
async def test_vector_store_clear() -> None:
    """Verify clear empties all contents of the store."""
    store = InMemoryVectorStore(dimension=2)
    c1 = _create_chunk(dense_vector=[1.0, 0.0])
    c2 = _create_chunk(dense_vector=[0.0, 1.0])

    await store.upsert([c1, c2])
    assert store.count() == 2

    await store.clear()
    assert store.count() == 0
    assert len(store) == 0


@pytest.mark.asyncio
async def test_vector_store_search_ordering_and_top_k() -> None:
    """Verify search orders by descending similarity score and respects top_k."""
    store = InMemoryVectorStore(dimension=2)

    # c1 is perfectly aligned with query [1.0, 0.0] -> score 1.0
    # c2 is at 45 degrees [1.0, 1.0] -> score ~0.707
    # c3 is orthogonal [0.0, 1.0] -> score 0.0
    # c4 is opposite [-1.0, 0.0] -> score -1.0
    c1 = _create_chunk(dense_vector=[1.0, 0.0], content="Perfect match")
    c2 = _create_chunk(dense_vector=[1.0, 1.0], content="Partial match")
    c3 = _create_chunk(dense_vector=[0.0, 1.0], content="Orthogonal")
    c4 = _create_chunk(dense_vector=[-1.0, 0.0], content="Opposite")

    await store.upsert([c1, c2, c3, c4])

    results = await store.search(query_vector=[1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk.id == c1.id
    assert results[0].score == pytest.approx(1.0)
    assert results[0].rank == 1
    assert results[0].retrieval_type == "dense"

    assert results[1].chunk.id == c2.id
    assert results[1].score == pytest.approx(0.7071, rel=1e-3)
    assert results[1].rank == 2

    # Scores strictly descending
    assert results[0].score >= results[1].score


@pytest.mark.asyncio
async def test_vector_store_metadata_and_provenance_preservation() -> None:
    """Verify chunk metadata and provenance details are preserved in search results."""
    store = InMemoryVectorStore(dimension=2)
    doc_id = uuid4()
    extra_meta = {"category": "architecture", "verified": True}

    chunk = _create_chunk(
        dense_vector=[1.0, 0.0],
        document_id=doc_id,
        content="Provenance tracking chunk",
        extra=extra_meta,
    )
    await store.upsert([chunk])

    results = await store.search(query_vector=[1.0, 0.0], top_k=1)
    retrieved_chunk = results[0].chunk

    assert retrieved_chunk.document_id == doc_id
    assert retrieved_chunk.chunk_index == 0
    assert retrieved_chunk.content == "Provenance tracking chunk"
    assert retrieved_chunk.metadata.section_header == "Introduction"
    assert retrieved_chunk.metadata.start_char_idx == 0
    assert retrieved_chunk.metadata.extra["category"] == "architecture"
    assert retrieved_chunk.metadata.extra["verified"] is True


@pytest.mark.asyncio
async def test_vector_store_filtering() -> None:
    """Verify search respects metadata filters."""
    store = InMemoryVectorStore(dimension=2)
    doc_a = uuid4()
    doc_b = uuid4()

    c1 = _create_chunk(
        dense_vector=[1.0, 0.0],
        document_id=doc_a,
        content="Doc A chunk",
        extra={"tier": "gold"},
    )
    c2 = _create_chunk(
        dense_vector=[1.0, 0.0],
        document_id=doc_b,
        content="Doc B chunk",
        extra={"tier": "silver"},
    )
    await store.upsert([c1, c2])

    results_doc_a = await store.search(
        query_vector=[1.0, 0.0],
        filters={"document_id": doc_a},
    )
    assert len(results_doc_a) == 1
    assert results_doc_a[0].chunk.id == c1.id

    results_tier = await store.search(
        query_vector=[1.0, 0.0],
        filters={"tier": "silver"},
    )
    assert len(results_tier) == 1
    assert results_tier[0].chunk.id == c2.id


@pytest.mark.asyncio
async def test_vector_store_empty_store_behavior() -> None:
    """Verify searching an empty vector store returns an empty list."""
    store = InMemoryVectorStore(dimension=3)
    results = await store.search(query_vector=[1.0, 2.0, 3.0], top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_vector_store_dimension_validation() -> None:
    """Verify store enforces vector dimension matching during upsert and search."""
    store = InMemoryVectorStore(dimension=3)

    invalid_dim_chunk = _create_chunk(dense_vector=[1.0, 2.0])  # Dim 2 instead of 3
    with pytest.raises(VectorDimensionMismatchError, match="does not match store"):
        await store.upsert([invalid_dim_chunk])

    valid_chunk = _create_chunk(dense_vector=[1.0, 2.0, 3.0])
    await store.upsert([valid_chunk])

    with pytest.raises(VectorDimensionMismatchError, match="does not match store dimension"):
        await store.search(query_vector=[1.0, 2.0])  # Dim 2 instead of 3


@pytest.mark.asyncio
async def test_vector_store_missing_dense_vector() -> None:
    """Verify upserting a chunk without dense_vector raises VectorStoreError."""
    store = InMemoryVectorStore(dimension=2)
    chunk_without_vec = _create_chunk(dense_vector=None)

    with pytest.raises(VectorStoreError, match="'dense_vector' is None"):
        await store.upsert([chunk_without_vec])


@pytest.mark.asyncio
async def test_vector_store_invalid_search_inputs() -> None:
    """Verify empty query vector and invalid top_k raise VectorStoreError."""
    store = InMemoryVectorStore(dimension=2)

    with pytest.raises(VectorStoreError, match="Query vector cannot be empty"):
        await store.search(query_vector=[], top_k=5)

    with pytest.raises(VectorStoreError, match="top_k must be a positive integer"):
        await store.search(query_vector=[1.0, 2.0], top_k=0)

    with pytest.raises(VectorStoreError, match="top_k must be a positive integer"):
        await store.search(query_vector=[1.0, 2.0], top_k=-5)


@pytest.mark.asyncio
async def test_vector_store_zero_vector_handling() -> None:
    """Verify storing and querying zero vectors does not crash the store."""
    store = InMemoryVectorStore(dimension=2)
    zero_chunk = _create_chunk(dense_vector=[0.0, 0.0])

    await store.upsert([zero_chunk])
    results = await store.search(query_vector=[0.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].score == 0.0
