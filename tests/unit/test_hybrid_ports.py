"""Unit tests for Stage 11 Step 1: Hybrid retrieval domain models and port abstractions."""

from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import pytest

from ragforge.domain.models import Chunk, ChunkMetadata, RetrievedChunk
from ragforge.ports import BaseLexicalIndex, BaseScoreFusion
from ragforge.ports.fusion import BaseScoreFusion as DirectBaseScoreFusion
from ragforge.ports.lexical import BaseLexicalIndex as DirectBaseLexicalIndex


def _create_sample_chunk() -> Chunk:
    """Helper creating a minimal valid Chunk."""
    return Chunk(
        id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content="Test content for hybrid retrieval.",
        token_count=6,
        content_hash="test-hash-123",
        metadata=ChunkMetadata(),
    )


class TestRetrievedChunkDomainModel:
    """Verify RetrievedChunk score preservation and backward compatibility."""

    def test_default_dense_and_lexical_scores_are_none(self) -> None:
        """Existing instantiation pattern without hybrid score fields remains valid and None."""
        chunk = _create_sample_chunk()
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.88,
            retrieval_type="dense",
            rank=1,
        )

        assert retrieved.score == 0.88
        assert retrieved.dense_score is None
        assert retrieved.lexical_score is None
        assert retrieved.retrieval_type == "dense"
        assert retrieved.rank == 1

    def test_preserves_dense_and_lexical_scores(self) -> None:
        """RetrievedChunk preserves explicit score, dense_score, and lexical_score."""
        chunk = _create_sample_chunk()
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.0325,  # fused RRF score
            retrieval_type="hybrid",
            rank=1,
            dense_score=0.912,
            lexical_score=14.35,
        )

        assert retrieved.score == 0.0325
        assert retrieved.dense_score == 0.912
        assert retrieved.lexical_score == 14.35
        assert retrieved.retrieval_type == "hybrid"
        assert retrieved.rank == 1

    def test_serialization_round_trip_preserves_scores(self) -> None:
        """RetrievedChunk serialization preserves dense and lexical score fields."""
        chunk = _create_sample_chunk()
        retrieved = RetrievedChunk(
            chunk=chunk,
            score=0.05,
            retrieval_type="hybrid",
            rank=2,
            dense_score=0.75,
            lexical_score=8.5,
        )

        dumped = retrieved.model_dump()
        assert dumped["score"] == 0.05
        assert dumped["dense_score"] == 0.75
        assert dumped["lexical_score"] == 8.5

        restored = RetrievedChunk.model_validate(dumped)
        assert restored.score == 0.05
        assert restored.dense_score == 0.75
        assert restored.lexical_score == 8.5


class TestPortExports:
    """Verify export contracts for the new hybrid retrieval ports."""

    def test_ports_are_importable_from_package(self) -> None:
        """BaseLexicalIndex and BaseScoreFusion are exported by ragforge.ports."""
        import ragforge.ports as ports_pkg

        assert "BaseLexicalIndex" in ports_pkg.__all__
        assert "BaseScoreFusion" in ports_pkg.__all__
        assert ports_pkg.BaseLexicalIndex is DirectBaseLexicalIndex
        assert ports_pkg.BaseScoreFusion is DirectBaseScoreFusion


class TestBaseLexicalIndexPort:
    """Verify abstract enforcement for BaseLexicalIndex."""

    def test_abstract_class_cannot_be_instantiated_directly(self) -> None:
        """BaseLexicalIndex cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class BaseLexicalIndex"):
            BaseLexicalIndex()  # type: ignore[abstract]

    def test_incomplete_subclass_cannot_be_instantiated(self) -> None:
        """Subclass missing any abstract method cannot be instantiated."""

        class IncompleteLexicalIndex(BaseLexicalIndex):
            async def index_chunks(self, chunks: Sequence[Chunk]) -> None:
                pass

        with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class IncompleteLexicalIndex",
        ):
            IncompleteLexicalIndex()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_concrete_subclass_implements_all_methods(self) -> None:
        """A complete concrete implementation satisfies the BaseLexicalIndex contract."""

        class ConcreteLexicalIndex(BaseLexicalIndex):
            def __init__(self) -> None:
                self._chunks: dict[UUID, Chunk] = {}

            async def index_chunks(self, chunks: Sequence[Chunk]) -> None:
                for c in chunks:
                    self._chunks[c.id] = c

            async def search(
                self,
                query: str,
                top_k: int = 10,
                filters: dict[str, Any] | None = None,
            ) -> list[RetrievedChunk]:
                return [
                    RetrievedChunk(
                        chunk=c,
                        score=1.0,
                        retrieval_type="lexical",
                        rank=i + 1,
                        lexical_score=1.0,
                    )
                    for i, c in enumerate(list(self._chunks.values())[:top_k])
                ]

            async def delete_by_document_id(self, document_id: UUID) -> None:
                self._chunks = {
                    cid: c for cid, c in self._chunks.items() if c.document_id != document_id
                }

            async def delete_by_chunk_id(self, chunk_id: UUID) -> bool:
                if chunk_id in self._chunks:
                    del self._chunks[chunk_id]
                    return True
                return False

            async def clear(self) -> None:
                self._chunks.clear()

            def count(self) -> int:
                return len(self._chunks)

        index = ConcreteLexicalIndex()
        assert index.count() == 0

        chunk = _create_sample_chunk()
        await index.index_chunks([chunk])
        assert index.count() == 1

        results = await index.search("test", top_k=5)
        assert len(results) == 1
        assert results[0].lexical_score == 1.0

        deleted = await index.delete_by_chunk_id(chunk.id)
        assert deleted is True
        assert index.count() == 0


class TestBaseScoreFusionPort:
    """Verify abstract enforcement for BaseScoreFusion."""

    def test_abstract_class_cannot_be_instantiated_directly(self) -> None:
        """BaseScoreFusion cannot be instantiated without implementing abstract methods."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class BaseScoreFusion"):
            BaseScoreFusion()  # type: ignore[abstract]

    def test_concrete_subclass_implements_fuse(self) -> None:
        """A concrete implementation satisfies the BaseScoreFusion contract."""

        class ConcreteScoreFusion(BaseScoreFusion):
            def fuse(
                self,
                dense_results: Sequence[RetrievedChunk],
                lexical_results: Sequence[RetrievedChunk],
                top_k: int | None = None,
            ) -> list[RetrievedChunk]:
                combined = list(dense_results) + list(lexical_results)
                return combined[:top_k] if top_k is not None else combined

        fusion = ConcreteScoreFusion()
        chunk1 = _create_sample_chunk()
        chunk2 = _create_sample_chunk()
        dense = [RetrievedChunk(chunk=chunk1, score=0.9, rank=1, dense_score=0.9)]
        lexical = [RetrievedChunk(chunk=chunk2, score=5.0, rank=1, lexical_score=5.0)]

        fused = fusion.fuse(dense_results=dense, lexical_results=lexical, top_k=1)
        assert len(fused) == 1
