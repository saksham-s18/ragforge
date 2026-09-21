from uuid import uuid4

import pytest

from ragforge.adapters.embeddings import DeterministicEmbeddingProvider
from ragforge.adapters.vector_stores import InMemoryVectorStore
from ragforge.domain.exceptions import VectorStoreError
from ragforge.domain.models import Chunk, ChunkMetadata, Query
from ragforge.services import RetrievalService


def _create_sample_chunks() -> list[Chunk]:
    """Helper to generate sample domain chunks."""
    doc_id = uuid4()
    contents = [
        "FastAPI framework handles asynchronous request routing and schema validation.",
        "Deterministic chunking preserves Markdown section headings and offsets.",
        "Astronomical observatories monitor distant galaxies and planetary nebulae.",
    ]
    return [
        Chunk(
            id=uuid4(),
            document_id=doc_id,
            chunk_index=i,
            content=text,
            token_count=12,
            content_hash=f"hash_{i}",
            metadata=ChunkMetadata(
                section_header=f"Section {i}",
                start_char_idx=0,
                end_char_idx=len(text),
                extra={"topic": "engineering" if i < 2 else "science"},
            ),
        )
        for i, text in enumerate(contents)
    ]


@pytest.mark.asyncio
async def test_retrieval_service_query_embedding_and_ordering() -> None:
    """Verify RetrievalService embeds query and returns results sorted by descending similarity."""
    provider = DeterministicEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    # Query clearly relevant to the first chunk
    query = "asynchronous request routing in FastAPI"
    results = await service.retrieve(query, top_k=3)

    assert len(results) == 3
    # First result should be the FastAPI chunk
    assert "FastAPI" in results[0].chunk.content
    assert results[0].rank == 1

    # Verify descending score ordering
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert results[i].rank == i + 1


@pytest.mark.asyncio
async def test_retrieval_service_query_domain_model() -> None:
    """Verify RetrievalService supports Query domain model."""
    provider = DeterministicEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    query_model = Query(
        raw_query="Markdown section headings chunking",
        top_k=2,
    )
    results = await service.retrieve(query_model)

    assert len(results) == 2
    assert "chunking" in results[0].chunk.content.lower()
    assert results[0].rank == 1
    assert results[1].rank == 2


@pytest.mark.asyncio
async def test_retrieval_service_metadata_propagation() -> None:
    """Verify full chunk metadata and document provenance survive retrieval."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    results = await service.retrieve("galaxies nebulae", top_k=1)
    assert len(results) == 1

    retrieved = results[0]
    assert retrieved.chunk.metadata.section_header == "Section 2"
    assert retrieved.chunk.metadata.extra["topic"] == "science"
    assert retrieved.chunk.document_id == chunks[2].document_id


@pytest.mark.asyncio
async def test_retrieval_service_invalid_top_k() -> None:
    """Verify invalid top_k values raise VectorStoreError."""
    provider = DeterministicEmbeddingProvider(dimension=16)
    store = InMemoryVectorStore(dimension=16)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    with pytest.raises(VectorStoreError, match="top_k must be a positive integer"):
        await service.retrieve("test query", top_k=0)

    with pytest.raises(VectorStoreError, match="top_k must be a positive integer"):
        await service.retrieve("test query", top_k=-3)


@pytest.mark.asyncio
async def test_retrieval_service_empty_vector_store() -> None:
    """Verify querying an empty vector store returns an empty list without error."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    results = await service.retrieve("Search on empty store", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_retrieval_service_empty_query() -> None:
    """Verify empty or whitespace query returns an empty result list immediately."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    results_empty = await service.retrieve("", top_k=5)
    results_ws = await service.retrieve("   \t \n  ", top_k=5)

    assert results_empty == []
    assert results_ws == []


@pytest.mark.asyncio
async def test_retrieval_service_score_threshold_filtering() -> None:
    """Verify score_threshold discards low-similarity candidates and re-indexes ranks."""
    provider = DeterministicEmbeddingProvider(dimension=64)
    store = InMemoryVectorStore(dimension=64)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    # Threshold set high enough to filter out distant chunks
    results = await service.retrieve(
        "FastAPI asynchronous",
        top_k=3,
        score_threshold=0.2,
    )

    for item in results:
        assert item.score >= 0.2
    for idx, item in enumerate(results, start=1):
        assert item.rank == idx
