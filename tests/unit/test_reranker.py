"""Unit tests for BaseReranker port, DeterministicReranker adapter, and ranking logic."""

from uuid import UUID, uuid4

import pytest

from ragforge.adapters.rerankers import DeterministicReranker
from ragforge.domain.models import Chunk, ChunkMetadata, RetrievedChunk
from ragforge.ports.reranker import BaseReranker


def make_test_chunk(
    content: str,
    score: float = 0.85,
    rank: int = 1,
    doc_id: UUID | None = None,
    chunk_id: UUID | None = None,
    section: str = "Intro",
) -> RetrievedChunk:
    """Helper creating a test RetrievedChunk with full metadata."""
    doc_uuid = doc_id or uuid4()
    c_uuid = chunk_id or uuid4()
    chunk = Chunk(
        id=c_uuid,
        document_id=doc_uuid,
        chunk_index=0,
        content=content,
        token_count=len(content.split()),
        content_hash="testhash",
        metadata=ChunkMetadata(
            section_header=section,
            page_number=1,
            start_char_idx=0,
            end_char_idx=len(content),
            extra={"source_path": "docs/test.md", "document_title": "Test Doc"},
        ),
    )
    return RetrievedChunk(
        chunk=chunk,
        score=score,
        retrieval_type="dense",
        rank=rank,
    )


# ---------------------------------------------------------------------------
# 1. Reranker Port & Interface Tests
# ---------------------------------------------------------------------------


def test_base_reranker_abstract_contract() -> None:
    """Verify that BaseReranker cannot be instantiated directly without implementing rerank."""
    with pytest.raises(TypeError):
        BaseReranker()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 2. Deterministic Reranker Initialization & Validation
# ---------------------------------------------------------------------------


def test_deterministic_reranker_init_defaults() -> None:
    """Verify default weights and configuration for DeterministicReranker."""
    reranker = DeterministicReranker()
    assert reranker.query_term_weight == 0.6
    assert reranker.term_frequency_weight == 0.2
    assert reranker.phrase_match_weight == 0.2


def test_deterministic_reranker_init_invalid_weights() -> None:
    """Verify ValueError is raised on negative or zero total weights."""
    with pytest.raises(ValueError, match="non-negative"):
        DeterministicReranker(query_term_weight=-0.1)

    with pytest.raises(ValueError, match="greater than 0"):
        DeterministicReranker(
            query_term_weight=0.0,
            term_frequency_weight=0.0,
            phrase_match_weight=0.0,
        )


# ---------------------------------------------------------------------------
# 3. Deterministic Reranker Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_reranker_empty_chunks() -> None:
    """Verify empty input candidates return an empty list."""
    reranker = DeterministicReranker()
    result = await reranker.rerank(query="database indexing", chunks=[], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_deterministic_reranker_empty_query() -> None:
    """Verify empty or whitespace query assigns zero score and preserves candidates."""
    reranker = DeterministicReranker()
    c1 = make_test_chunk("First chunk content", score=0.9, rank=1)
    c2 = make_test_chunk("Second chunk content", score=0.8, rank=2)

    result = await reranker.rerank(query="   ", chunks=[c1, c2], top_k=2)
    assert len(result) == 2
    assert result[0].rerank_score == 0.0
    assert result[1].rerank_score == 0.0
    assert result[0].rank == 1
    assert result[1].rank == 2


@pytest.mark.asyncio
async def test_deterministic_reranker_invalid_top_k() -> None:
    """Verify invalid top_k values raise ValueError."""
    reranker = DeterministicReranker()
    c1 = make_test_chunk("Some content")
    with pytest.raises(ValueError, match="positive integer"):
        await reranker.rerank(query="test", chunks=[c1], top_k=0)
    with pytest.raises(ValueError, match="positive integer"):
        await reranker.rerank(query="test", chunks=[c1], top_k=-5)


# ---------------------------------------------------------------------------
# 4. Lexical Scoring & Candidate Reordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_reranker_reorders_by_lexical_relevance() -> None:
    """Verify that a candidate with lower vector similarity but higher lexical match
    is reordered to rank #1, while preserving original retrieval score."""
    reranker = DeterministicReranker()

    # Candidate A: high vector score, but vague semantic match without exact query terms
    chunk_a = make_test_chunk(
        content="General storage systems handle vast amounts of unstructured information.",
        score=0.92,
        rank=1,
    )
    # Candidate B: lower vector score, but exact keyword match for the user query
    chunk_b = make_test_chunk(
        content="Qdrant vector database provides persistent indexing and storage for embeddings.",
        score=0.78,
        rank=2,
    )

    query = "Qdrant vector database"
    reranked = await reranker.rerank(query=query, chunks=[chunk_a, chunk_b])

    assert len(reranked) == 2
    # Chunk B should now be rank 1
    assert reranked[0].chunk.id == chunk_b.chunk.id
    assert reranked[0].rank == 1
    assert reranked[0].score == 0.78  # Original score strictly preserved!
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > 0.5

    # Chunk A should now be rank 2
    assert reranked[1].chunk.id == chunk_a.chunk.id
    assert reranked[1].rank == 2
    assert reranked[1].score == 0.92  # Original score strictly preserved!
    assert reranked[1].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score


@pytest.mark.asyncio
async def test_deterministic_reranker_exact_phrase_bonus() -> None:
    """Verify that chunks with exact contiguous phrase matches score higher than
    chunks with scattered keywords."""
    reranker = DeterministicReranker()

    # Chunk 1 has exact contiguous phrase
    c1 = make_test_chunk(
        content="RAGForge supports deterministic chunking out of the box.",
        score=0.80,
        rank=1,
    )
    # Chunk 2 has keywords scattered separately
    c2 = make_test_chunk(
        content="The system supports chunking, and the deterministic mode is optional.",
        score=0.80,
        rank=2,
    )

    query = "deterministic chunking"
    reranked = await reranker.rerank(query=query, chunks=[c2, c1])

    assert reranked[0].chunk.id == c1.chunk.id
    assert reranked[0].rank == 1
    assert reranked[0].rerank_score is not None
    assert reranked[1].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score


# ---------------------------------------------------------------------------
# 5. Top-K Selection & Slicing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_reranker_top_k_selection() -> None:
    """Verify that top_k cuts off candidates after reranking."""
    reranker = DeterministicReranker()
    candidates = [
        make_test_chunk(
            f"Candidate chunk number {i} discussing topic {i}",
            score=0.5 + i * 0.04,
            rank=i,
        )
        for i in range(1, 11)
    ]

    # Rerank 10 candidates with top_k=3
    reranked = await reranker.rerank(query="topic 8", chunks=candidates, top_k=3)

    assert len(reranked) == 3
    assert reranked[0].rank == 1
    assert reranked[1].rank == 2
    assert reranked[2].rank == 3
    # Candidate 8 matches both "topic" and "8" and should be rank 1
    assert "topic 8" in reranked[0].chunk.content


@pytest.mark.asyncio
async def test_deterministic_reranker_top_k_larger_than_pool() -> None:
    """Verify when top_k exceeds candidate pool, all candidates are returned ranked."""
    reranker = DeterministicReranker()
    c1 = make_test_chunk("Content A", rank=1)
    c2 = make_test_chunk("Content B", rank=2)

    reranked = await reranker.rerank(query="Content", chunks=[c1, c2], top_k=10)
    assert len(reranked) == 2
    assert [r.rank for r in reranked] == [1, 2]


# ---------------------------------------------------------------------------
# 6. Provenance & Attribution Preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_reranker_preserves_all_provenance() -> None:
    """Verify that chunk metadata, offsets, documents, and original scores are intact."""
    reranker = DeterministicReranker()
    doc_id = uuid4()
    chunk_id = uuid4()
    original = make_test_chunk(
        content="Retrieval Augmented Generation combines neural search with LLM synthesis.",
        score=0.8877,
        rank=1,
        doc_id=doc_id,
        chunk_id=chunk_id,
        section="Architecture Overview",
    )

    reranked = await reranker.rerank(query="neural search", chunks=[original])
    assert len(reranked) == 1
    item = reranked[0]

    # Verify chunk attributes
    assert item.chunk.id == chunk_id
    assert item.chunk.document_id == doc_id
    assert item.chunk.content == original.chunk.content
    assert item.chunk.metadata.section_header == "Architecture Overview"
    assert item.chunk.metadata.page_number == 1
    assert item.chunk.metadata.start_char_idx == 0
    assert item.chunk.metadata.end_char_idx == len(original.chunk.content)
    assert item.chunk.metadata.extra["source_path"] == "docs/test.md"

    # Verify scores and ranks
    assert item.score == 0.8877  # Vector score preserved
    assert item.rerank_score is not None
    assert item.rerank_score > 0.0
    assert item.rank == 1
    assert item.retrieval_type == "dense"


# ---------------------------------------------------------------------------
# 7. Determinism & Tie-Breaking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_reranker_reproducible_tie_breaking() -> None:
    """Verify that chunks with identical lexical scores break ties deterministically
    via initial score, initial rank, and chunk ID."""
    reranker = DeterministicReranker()

    c1 = make_test_chunk("Identical content text", score=0.90, rank=1)
    c2 = make_test_chunk("Identical content text", score=0.80, rank=2)

    # c1 has higher vector score -> ranks first
    res1 = await reranker.rerank(query="Identical text", chunks=[c2, c1])
    assert res1[0].chunk.id == c1.chunk.id
    assert res1[1].chunk.id == c2.chunk.id

    # Running again yields identical result
    res2 = await reranker.rerank(query="Identical text", chunks=[c1, c2])
    assert res2[0].chunk.id == c1.chunk.id
    assert res2[1].chunk.id == c2.chunk.id
