from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ragforge.adapters.embeddings import DeterministicEmbeddingProvider
from ragforge.adapters.fusion import ReciprocalRankFusion
from ragforge.adapters.lexical import BM25LexicalIndex
from ragforge.adapters.vector_stores import InMemoryVectorStore
from ragforge.domain.enums import SearchStrategy
from ragforge.domain.exceptions import VectorStoreError
from ragforge.domain.models import Chunk, ChunkMetadata, Query, RetrievedChunk
from ragforge.ports.fusion import BaseScoreFusion
from ragforge.ports.lexical import BaseLexicalIndex
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


# ---------------------------------------------------------------------------
# Stage 11 Step 3: Hybrid and Sparse Retrieval Tests (1 to 19)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_service_default_strategy_is_dense() -> None:
    """1. Verify default behavior remains DENSE with dense scores and None lexical scores."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    results = await service.retrieve("FastAPI framework")
    assert len(results) > 0
    for r in results:
        assert r.retrieval_type == "dense"
        assert r.dense_score == r.score
        assert r.lexical_score is None


@pytest.mark.asyncio
async def test_retrieval_service_dense_does_not_call_lexical_index() -> None:
    """2. Verify dense retrieval does not call the lexical index."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    mock_lexical = AsyncMock(spec=BaseLexicalIndex)
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=mock_lexical,
    )

    chunks = _create_sample_chunks()
    # index into store directly to avoid calling mock_lexical during setup
    await store.upsert(
        [
            Chunk(**{**c.model_dump(), "dense_vector": await provider.embed_query(c.content)})
            for c in chunks
        ]
    )

    results = await service.retrieve("FastAPI framework", search_strategy=SearchStrategy.DENSE)
    assert len(results) > 0
    mock_lexical.search.assert_not_called()


@pytest.mark.asyncio
async def test_retrieval_service_sparse_calls_lexical_index() -> None:
    """3. Verify sparse retrieval queries the lexical index and returns lexical scores."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
    )

    chunks = _create_sample_chunks()
    await lexical.index_chunks(chunks)

    results = await service.retrieve("FastAPI", search_strategy=SearchStrategy.SPARSE)
    assert len(results) > 0
    assert results[0].retrieval_type == "lexical"
    assert results[0].lexical_score is not None
    assert results[0].dense_score is None
    assert results[0].score == results[0].lexical_score


@pytest.mark.asyncio
async def test_retrieval_service_hybrid_calls_both_dense_and_lexical() -> None:
    """4. Verify hybrid retrieval queries both vector store and lexical index."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    mock_store = AsyncMock(spec=InMemoryVectorStore)
    mock_lexical = AsyncMock(spec=BaseLexicalIndex)
    mock_fusion = MagicMock(spec=BaseScoreFusion)

    chunks = _create_sample_chunks()
    dummy_dense = [RetrievedChunk(chunk=chunks[0], score=0.9, retrieval_type="dense", rank=1)]
    dummy_lexical = [RetrievedChunk(chunk=chunks[0], score=5.0, retrieval_type="lexical", rank=1)]

    mock_store.search.return_value = dummy_dense
    mock_lexical.search.return_value = dummy_lexical
    mock_fusion.fuse.return_value = [
        RetrievedChunk(
            chunk=chunks[0],
            score=0.03,
            retrieval_type="hybrid",
            rank=1,
            dense_score=0.9,
            lexical_score=5.0,
        )
    ]

    service = RetrievalService(
        embedding_provider=provider,
        vector_store=mock_store,
        lexical_index=mock_lexical,
        fusion=mock_fusion,
    )

    results = await service.retrieve("FastAPI", search_strategy=SearchStrategy.HYBRID)
    assert len(results) == 1
    mock_store.search.assert_called_once()
    mock_lexical.search.assert_called_once()


@pytest.mark.asyncio
async def test_retrieval_service_hybrid_passes_results_through_rrf() -> None:
    """5. Verify hybrid candidates are passed to the score fusion adapter."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    mock_fusion = MagicMock(spec=ReciprocalRankFusion)
    mock_fusion.fuse.return_value = []

    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=mock_fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    await service.retrieve("FastAPI", top_k=2, search_strategy=SearchStrategy.HYBRID)
    assert mock_fusion.fuse.called
    kwargs = mock_fusion.fuse.call_args.kwargs
    assert "dense_results" in kwargs
    assert "lexical_results" in kwargs
    assert kwargs["top_k"] == 2


@pytest.mark.asyncio
async def test_retrieval_service_fused_score_in_score() -> None:
    """6. Verify fused RRF score is stored in `score`."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion(k=60)
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    results = await service.retrieve(
        "FastAPI asynchronous", top_k=1, search_strategy=SearchStrategy.HYBRID
    )
    assert len(results) == 1
    # Ranked #1 in both channels -> score = 1/(60+1) + 1/(60+1) = 2/61 ~ 0.032786
    expected_score = round(2.0 / 61.0, 6)
    assert round(results[0].score, 6) == expected_score
    assert results[0].retrieval_type == "hybrid"


@pytest.mark.asyncio
async def test_retrieval_service_dense_score_preserved() -> None:
    """7. Verify original dense score is preserved in `dense_score`."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    results = await service.retrieve(
        "FastAPI asynchronous", top_k=1, search_strategy=SearchStrategy.HYBRID
    )
    assert len(results) == 1
    assert results[0].dense_score is not None
    assert results[0].dense_score != results[0].score
    assert -1.0 <= results[0].dense_score <= 1.0


@pytest.mark.asyncio
async def test_retrieval_service_lexical_score_preserved() -> None:
    """8. Verify original BM25 score is preserved in `lexical_score`."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    results = await service.retrieve(
        "FastAPI asynchronous", top_k=1, search_strategy=SearchStrategy.HYBRID
    )
    assert len(results) == 1
    assert results[0].lexical_score is not None
    assert results[0].lexical_score != results[0].score
    assert results[0].lexical_score > 0.0


@pytest.mark.asyncio
async def test_retrieval_service_provenance_survives_fusion() -> None:
    """9. Verify chunk metadata and provenance survive hybrid fusion."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    results = await service.retrieve(
        "observatories galaxies", search_strategy=SearchStrategy.HYBRID
    )
    assert len(results) > 0
    top = results[0]
    assert top.chunk.id == chunks[2].id
    assert top.chunk.document_id == chunks[2].document_id
    assert top.chunk.metadata.section_header == "Section 2"
    assert top.chunk.metadata.extra["topic"] == "science"


@pytest.mark.asyncio
async def test_retrieval_service_empty_lexical_falls_back_to_dense() -> None:
    """10. Verify empty lexical results fall back to dense results."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()  # Empty lexical index
    fusion = ReciprocalRankFusion()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    # Index only into vector store
    for c in chunks:
        c.dense_vector = await provider.embed_query(c.content)
    await store.upsert(chunks)

    results = await service.retrieve("FastAPI framework", search_strategy=SearchStrategy.HYBRID)
    assert len(results) > 0
    assert results[0].retrieval_type == "dense"
    assert results[0].dense_score is not None
    assert results[0].lexical_score is None


@pytest.mark.asyncio
async def test_retrieval_service_empty_dense_falls_back_to_lexical() -> None:
    """11. Verify empty dense results fall back to lexical results."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)  # Empty vector store
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await lexical.index_chunks(chunks)

    results = await service.retrieve("FastAPI framework", search_strategy=SearchStrategy.HYBRID)
    assert len(results) > 0
    assert results[0].retrieval_type == "lexical"
    assert results[0].lexical_score is not None
    assert results[0].dense_score is None


@pytest.mark.asyncio
async def test_retrieval_service_lexical_failure_falls_back_to_dense() -> None:
    """12. Verify lexical channel failure in HYBRID falls back to dense results."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    mock_lexical = AsyncMock(spec=BaseLexicalIndex)
    mock_lexical.search.side_effect = RuntimeError("BM25 internal corruption")
    fusion = ReciprocalRankFusion()

    chunks = _create_sample_chunks()
    for c in chunks:
        c.dense_vector = await provider.embed_query(c.content)
    await store.upsert(chunks)

    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=mock_lexical,
        fusion=fusion,
    )

    results = await service.retrieve("FastAPI framework", search_strategy=SearchStrategy.HYBRID)
    assert len(results) > 0
    assert results[0].retrieval_type == "dense"
    assert results[0].dense_score is not None


@pytest.mark.asyncio
async def test_retrieval_service_dense_failure_falls_back_to_lexical() -> None:
    """13. Verify dense channel failure in HYBRID falls back to lexical results."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    mock_store = AsyncMock(spec=InMemoryVectorStore)
    mock_store.search.side_effect = VectorStoreError("Vector store unavailable")
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()

    chunks = _create_sample_chunks()
    await lexical.index_chunks(chunks)

    service = RetrievalService(
        embedding_provider=provider,
        vector_store=mock_store,
        lexical_index=lexical,
        fusion=fusion,
    )

    results = await service.retrieve("FastAPI framework", search_strategy=SearchStrategy.HYBRID)
    assert len(results) > 0
    assert results[0].retrieval_type == "lexical"
    assert results[0].lexical_score is not None


@pytest.mark.asyncio
async def test_retrieval_service_both_failures_raise_vector_store_error() -> None:
    """14. Verify failure of both channels raises VectorStoreError."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    mock_store = AsyncMock(spec=InMemoryVectorStore)
    mock_store.search.side_effect = VectorStoreError("Vector store down")
    mock_lexical = AsyncMock(spec=BaseLexicalIndex)
    mock_lexical.search.side_effect = RuntimeError("Lexical index failed")
    fusion = ReciprocalRankFusion()

    service = RetrievalService(
        embedding_provider=provider,
        vector_store=mock_store,
        lexical_index=mock_lexical,
        fusion=fusion,
    )

    with pytest.raises(VectorStoreError, match="Both retrieval channels failed"):
        await service.retrieve("FastAPI", search_strategy=SearchStrategy.HYBRID)


@pytest.mark.asyncio
async def test_retrieval_service_missing_lexical_dependency_in_hybrid() -> None:
    """15. Verify missing lexical index in HYBRID fails clearly."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    fusion = ReciprocalRankFusion()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=None,
        fusion=fusion,
    )

    with pytest.raises(ValueError, match="Lexical index is required for HYBRID"):
        await service.retrieve("FastAPI", search_strategy=SearchStrategy.HYBRID)


@pytest.mark.asyncio
async def test_retrieval_service_missing_fusion_dependency_in_hybrid() -> None:
    """16. Verify missing fusion dependency in HYBRID fails clearly."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=None,
    )

    with pytest.raises(ValueError, match="Score fusion is required for HYBRID"):
        await service.retrieve("FastAPI", search_strategy=SearchStrategy.HYBRID)


@pytest.mark.asyncio
async def test_retrieval_service_dense_threshold_remains_dense_only() -> None:
    """17. Verify score threshold applies strictly to the dense retrieval channel."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()

    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    # Threshold 0.99 filters out all dense candidates, but BM25 produces lexical results
    results = await service.retrieve(
        "FastAPI framework",
        score_threshold=0.99,
        search_strategy=SearchStrategy.HYBRID,
    )
    # Since dense is empty due to high threshold, it falls back to lexical results
    assert len(results) > 0
    assert results[0].retrieval_type == "lexical"
    assert "FastAPI" in results[0].chunk.content


@pytest.mark.asyncio
async def test_retrieval_service_filters_propagate_to_both_channels() -> None:
    """18. Verify metadata filters are respected in hybrid retrieval."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()

    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    results = await service.retrieve(
        "headings astronomy galaxies",
        filters={"topic": "science"},
        search_strategy=SearchStrategy.HYBRID,
    )
    assert len(results) == 1
    assert results[0].chunk.metadata.extra["topic"] == "science"


@pytest.mark.asyncio
async def test_retrieval_service_deterministic_ordering_preserved() -> None:
    """19. Verify deterministic result ranking in hybrid search."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    store = InMemoryVectorStore(dimension=32)
    lexical = BM25LexicalIndex()
    fusion = ReciprocalRankFusion()

    service = RetrievalService(
        embedding_provider=provider,
        vector_store=store,
        lexical_index=lexical,
        fusion=fusion,
    )

    chunks = _create_sample_chunks()
    await service.index_chunks(chunks)

    res1 = await service.retrieve("Deterministic Markdown", search_strategy=SearchStrategy.HYBRID)
    res2 = await service.retrieve("Deterministic Markdown", search_strategy=SearchStrategy.HYBRID)

    assert len(res1) == len(res2)
    for r1, r2 in zip(res1, res2, strict=True):
        assert r1.chunk.id == r2.chunk.id
        assert r1.score == r2.score
        assert r1.rank == r2.rank
