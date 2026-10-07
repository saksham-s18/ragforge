"""Unit tests for ReciprocalRankFusion (RRF) score fusion adapter."""

from uuid import UUID, uuid4

import pytest

from ragforge.adapters.fusion import ReciprocalRankFusion
from ragforge.domain.models import Chunk, ChunkMetadata, RetrievedChunk


def _create_chunk(content: str = "sample text", chunk_id: UUID | None = None) -> Chunk:
    """Helper creating a domain Chunk instance for fusion tests."""
    c_id = chunk_id or uuid4()
    return Chunk(
        id=c_id,
        document_id=uuid4(),
        chunk_index=0,
        content=content,
        token_count=len(content.split()),
        content_hash=f"hash-{c_id}",
        metadata=ChunkMetadata(),
    )


def _create_retrieved(
    chunk: Chunk,
    score: float,
    retrieval_type: str = "dense",
    rank: int = 1,
    dense_score: float | None = None,
    lexical_score: float | None = None,
) -> RetrievedChunk:
    """Helper creating a RetrievedChunk with specified scores and rank."""
    d_score = (
        dense_score if dense_score is not None else (score if retrieval_type == "dense" else None)
    )
    l_score = (
        lexical_score
        if lexical_score is not None
        else (score if retrieval_type == "lexical" else None)
    )
    return RetrievedChunk(
        chunk=chunk,
        score=score,
        retrieval_type=retrieval_type,
        rank=rank,
        dense_score=d_score,
        lexical_score=l_score,
    )


class TestReciprocalRankFusion:
    """Test suite verifying ReciprocalRankFusion behavior, calculations, and contracts."""

    def test_invalid_init_parameters(self) -> None:
        """Constructor validates k and channel weight parameters."""
        with pytest.raises(ValueError, match="RRF constant k must be a positive integer"):
            ReciprocalRankFusion(k=0)

        with pytest.raises(ValueError, match="RRF constant k must be a positive integer"):
            ReciprocalRankFusion(k=-5)

        with pytest.raises(ValueError, match="dense_weight cannot be negative"):
            ReciprocalRankFusion(dense_weight=-0.1)

        with pytest.raises(ValueError, match="lexical_weight cannot be negative"):
            ReciprocalRankFusion(lexical_weight=-1.0)

    def test_both_lists_empty(self) -> None:
        """Fusing empty dense and lexical results returns an empty list."""
        fusion = ReciprocalRankFusion()
        assert fusion.fuse(dense_results=[], lexical_results=[]) == []

    def test_empty_dense_list(self) -> None:
        """Fusing with empty dense results scores only lexical candidates."""
        fusion = ReciprocalRankFusion(k=60)
        c1 = _create_chunk("lexical only")
        lexical = [_create_retrieved(c1, score=12.5, retrieval_type="lexical", rank=1)]

        fused = fusion.fuse(dense_results=[], lexical_results=lexical)
        assert len(fused) == 1
        assert fused[0].chunk.id == c1.id
        assert fused[0].score == pytest.approx(1.0 / (60 + 1))
        assert fused[0].dense_score is None
        assert fused[0].lexical_score == 12.5
        assert fused[0].retrieval_type == "hybrid"
        assert fused[0].rank == 1

    def test_empty_lexical_list(self) -> None:
        """Fusing with empty lexical results scores only dense candidates."""
        fusion = ReciprocalRankFusion(k=60)
        c1 = _create_chunk("dense only")
        dense = [_create_retrieved(c1, score=0.92, retrieval_type="dense", rank=1)]

        fused = fusion.fuse(dense_results=dense, lexical_results=[])
        assert len(fused) == 1
        assert fused[0].chunk.id == c1.id
        assert fused[0].score == pytest.approx(1.0 / (60 + 1))
        assert fused[0].dense_score == 0.92
        assert fused[0].lexical_score is None
        assert fused[0].retrieval_type == "hybrid"
        assert fused[0].rank == 1

    def test_manual_rrf_calculation_and_formula(self) -> None:
        """Numerically verify exact RRF formula: 1/(60+1) + 1/(60+1) = 2/61."""
        fusion = ReciprocalRankFusion(k=60, dense_weight=1.0, lexical_weight=1.0)
        c1 = _create_chunk("overlapping candidate")
        dense = [_create_retrieved(c1, score=0.95, retrieval_type="dense", rank=1)]
        lexical = [_create_retrieved(c1, score=15.0, retrieval_type="lexical", rank=1)]

        fused = fusion.fuse(dense_results=dense, lexical_results=lexical)
        assert len(fused) == 1
        expected_score = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert expected_score == pytest.approx(2.0 / 61.0)
        assert fused[0].score == pytest.approx(expected_score)
        assert fused[0].dense_score == 0.95
        assert fused[0].lexical_score == 15.0
        assert fused[0].retrieval_type == "hybrid"
        assert fused[0].rank == 1

    def test_custom_k(self) -> None:
        """Custom k alters reciprocal rank denominator: 1/(10+1) + 1/(10+1) = 2/11."""
        fusion = ReciprocalRankFusion(k=10)
        c1 = _create_chunk("k test")
        dense = [_create_retrieved(c1, score=0.8, rank=1)]
        lexical = [_create_retrieved(c1, score=10.0, rank=1)]

        fused = fusion.fuse(dense_results=dense, lexical_results=lexical)
        expected = 2.0 / 11.0
        assert fused[0].score == pytest.approx(expected)

    def test_custom_dense_and_lexical_weights(self) -> None:
        """Custom channel weights apply to numerator: (2.0/61) + (0.5/61) = 2.5/61."""
        fusion = ReciprocalRankFusion(k=60, dense_weight=2.0, lexical_weight=0.5)
        c1 = _create_chunk("weight test")
        dense = [_create_retrieved(c1, score=0.9, rank=1)]
        lexical = [_create_retrieved(c1, score=8.0, rank=1)]

        fused = fusion.fuse(dense_results=dense, lexical_results=lexical)
        expected = (2.0 / 61.0) + (0.5 / 61.0)
        assert fused[0].score == pytest.approx(expected)

    def test_overlapping_and_disjoint_candidates_no_duplicates(self) -> None:
        """Overlapping candidates merge into one result; disjoint candidates retain single score."""
        fusion = ReciprocalRankFusion(k=60)
        c_overlap = _create_chunk("overlap")
        c_dense_only = _create_chunk("dense only")
        c_lex_only = _create_chunk("lexical only")

        dense = [
            _create_retrieved(c_overlap, score=0.9, rank=1),
            _create_retrieved(c_dense_only, score=0.8, rank=2),
        ]
        lexical = [
            _create_retrieved(c_overlap, score=14.0, rank=1),
            _create_retrieved(c_lex_only, score=11.0, rank=2),
        ]

        fused = fusion.fuse(dense_results=dense, lexical_results=lexical)

        # 3 unique chunks, no duplicates
        assert len(fused) == 3
        chunk_ids = [r.chunk.id for r in fused]
        assert len(set(chunk_ids)) == 3
        assert chunk_ids[0] == c_overlap.id

        # Rank 1: overlap receives both contributions: 1/61 + 1/61 = 2/61
        assert fused[0].score == pytest.approx(2.0 / 61.0)
        assert fused[0].dense_score == 0.9
        assert fused[0].lexical_score == 14.0
        assert fused[0].rank == 1

        # Ranks 2 and 3: each receives 1/62
        assert fused[1].score == pytest.approx(1.0 / 62.0)
        assert fused[2].score == pytest.approx(1.0 / 62.0)

    def test_multiple_overlapping_candidates_ranking(self) -> None:
        """Multi-candidate rankings reflect relative channel placements."""
        fusion = ReciprocalRankFusion(k=60)
        c1 = _create_chunk("c1")
        c2 = _create_chunk("c2")
        c3 = _create_chunk("c3")

        # c1: rank 1 dense, rank 2 lexical -> 1/61 + 1/62
        # c2: rank 2 dense, rank 1 lexical -> 1/62 + 1/61 (same as c1)
        # c3: rank 3 dense, rank 3 lexical -> 1/63 + 1/63
        dense = [
            _create_retrieved(c1, score=0.95, rank=1),
            _create_retrieved(c2, score=0.85, rank=2),
            _create_retrieved(c3, score=0.75, rank=3),
        ]
        lexical = [
            _create_retrieved(c2, score=20.0, rank=1),
            _create_retrieved(c1, score=18.0, rank=2),
            _create_retrieved(c3, score=15.0, rank=3),
        ]

        fused = fusion.fuse(dense_results=dense, lexical_results=lexical)
        assert len(fused) == 3
        assert fused[0].score == pytest.approx(1.0 / 61.0 + 1.0 / 62.0)
        assert fused[1].score == pytest.approx(1.0 / 61.0 + 1.0 / 62.0)
        assert fused[2].score == pytest.approx(2.0 / 63.0)
        assert fused[2].chunk.id == c3.id

    def test_deterministic_ordering_tie_breaking(self) -> None:
        """Candidates with identical RRF scores tie-break deterministically by string chunk ID."""
        fusion = ReciprocalRankFusion(k=60)
        id_a = UUID("00000000-0000-0000-0000-000000000001")
        id_b = UUID("00000000-0000-0000-0000-000000000002")

        c_a = _create_chunk("a", chunk_id=id_a)
        c_b = _create_chunk("b", chunk_id=id_b)

        # Dense: c_b at rank 1, c_a at rank 2
        # Lexical: c_a at rank 1, c_b at rank 2
        # Both get 1/61 + 1/62 (identical score)
        dense = [
            _create_retrieved(c_b, score=0.9, rank=1),
            _create_retrieved(c_a, score=0.8, rank=2),
        ]
        lexical = [
            _create_retrieved(c_a, score=10.0, rank=1),
            _create_retrieved(c_b, score=9.0, rank=2),
        ]

        fused = fusion.fuse(dense_results=dense, lexical_results=lexical)
        assert fused[0].score == fused[1].score
        # Tie break: id_a < id_b lexicographically
        assert fused[0].chunk.id == id_a
        assert fused[1].chunk.id == id_b
        assert fused[0].rank == 1
        assert fused[1].rank == 2

    def test_top_k_pruning(self) -> None:
        """top_k returns at most top_k items with continuous ranks; None returns all."""
        fusion = ReciprocalRankFusion(k=60)
        chunks = [_create_chunk(f"c{i}") for i in range(10)]
        dense = [
            _create_retrieved(c, score=1.0 - i * 0.05, rank=i + 1) for i, c in enumerate(chunks)
        ]

        # top_k = 3
        fused_3 = fusion.fuse(dense_results=dense, lexical_results=[], top_k=3)
        assert len(fused_3) == 3
        assert [r.rank for r in fused_3] == [1, 2, 3]

        # top_k = None returns all 10
        fused_all = fusion.fuse(dense_results=dense, lexical_results=[], top_k=None)
        assert len(fused_all) == 10
        assert [r.rank for r in fused_all] == list(range(1, 11))

        # Invalid top_k
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            fusion.fuse(dense_results=dense, lexical_results=[], top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            fusion.fuse(dense_results=dense, lexical_results=[], top_k=-1)

    def test_preservation_of_dense_and_lexical_scores(self) -> None:
        """Candidate sub-scores are faithfully preserved in fused RetrievedChunk."""
        fusion = ReciprocalRankFusion(k=60)
        c1 = _create_chunk("score test")
        dense = [
            RetrievedChunk(
                chunk=c1,
                score=0.8765,
                retrieval_type="dense",
                rank=1,
                dense_score=0.8765,
                lexical_score=None,
            )
        ]
        lexical = [
            RetrievedChunk(
                chunk=c1,
                score=14.321,
                retrieval_type="lexical",
                rank=1,
                dense_score=None,
                lexical_score=14.321,
            )
        ]

        fused = fusion.fuse(dense_results=dense, lexical_results=lexical)
        assert len(fused) == 1
        assert fused[0].dense_score == 0.8765
        assert fused[0].lexical_score == 14.321
        assert fused[0].score == pytest.approx(2.0 / 61.0)
        assert fused[0].retrieval_type == "hybrid"
