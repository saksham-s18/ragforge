"""Unit tests for Okapi BM25 lexical search index adapter."""

from uuid import UUID, uuid4

import pytest

from ragforge.adapters.lexical import BM25LexicalIndex, tokenize
from ragforge.domain.models import Chunk, ChunkMetadata


def _create_chunk(
    content: str,
    doc_id: UUID | None = None,
    chunk_id: UUID | None = None,
    extra: dict | None = None,
) -> Chunk:
    """Helper creating a domain Chunk instance for lexical indexing tests."""
    c_id = chunk_id or uuid4()
    d_id = doc_id or uuid4()
    metadata = ChunkMetadata(extra=extra or {})
    return Chunk(
        id=c_id,
        document_id=d_id,
        chunk_index=0,
        content=content,
        token_count=len(content.split()),
        content_hash=f"hash-{c_id}",
        metadata=metadata,
    )


class TestBM25LexicalIndex:
    """Test suite verifying BM25LexicalIndex indexing, scoring, filtering, and lifecycle."""

    def test_invalid_init_parameters(self) -> None:
        """Verify constructor validates k1 and b parameters."""
        with pytest.raises(ValueError, match="k1 cannot be negative"):
            BM25LexicalIndex(k1=-0.5)

        with pytest.raises(ValueError, match="b must be between 0.0 and 1.0"):
            BM25LexicalIndex(b=-0.1)

        with pytest.raises(ValueError, match="b must be between 0.0 and 1.0"):
            BM25LexicalIndex(b=1.2)

    @pytest.mark.asyncio
    async def test_empty_index_returns_empty_list(self) -> None:
        """Searching an unpopulated index returns an empty list without error."""
        index = BM25LexicalIndex()
        assert index.count() == 0
        assert len(index) == 0

        results = await index.search("python programming", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_single_document_exact_match(self) -> None:
        """Indexing a single chunk and querying an exact term retrieves it with correct schema."""
        index = BM25LexicalIndex()
        chunk = _create_chunk("Deep learning architectures require massive training datasets.")
        await index.index_chunks([chunk])

        assert index.count() == 1
        stored = await index.get(chunk.id)
        assert stored is not None
        assert stored.id == chunk.id

        results = await index.search("architectures", top_k=10)
        assert len(results) == 1
        item = results[0]
        assert item.chunk.id == chunk.id
        assert item.score > 0.0
        assert item.lexical_score == item.score
        assert item.dense_score is None
        assert item.retrieval_type == "lexical"
        assert item.rank == 1

    @pytest.mark.asyncio
    async def test_multi_term_query_ranking(self) -> None:
        """Multi-term queries reward documents matching more query terms."""
        index = BM25LexicalIndex()
        c1 = _create_chunk("neural network optimization methods")
        c2 = _create_chunk("database query optimization techniques")
        c3 = _create_chunk("unrelated cooking recipes for dinner")
        await index.index_chunks([c1, c2, c3])

        results = await index.search("neural network optimization", top_k=5)
        assert len(results) == 2
        # c1 matches all 3 terms; c2 matches only 1 term ("optimization")
        assert results[0].chunk.id == c1.id
        assert results[1].chunk.id == c2.id
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_case_normalization(self) -> None:
        """Query and index terms match regardless of casing."""
        index = BM25LexicalIndex()
        chunk = _create_chunk("Python and Rust provide high performance.")
        await index.index_chunks([chunk])

        res_lower = await index.search("python", top_k=5)
        res_upper = await index.search("PYTHON", top_k=5)
        res_mixed = await index.search("PyThOn", top_k=5)

        assert len(res_lower) == 1
        assert len(res_upper) == 1
        assert len(res_mixed) == 1
        assert res_lower[0].score == pytest.approx(res_upper[0].score)
        assert res_lower[0].score == pytest.approx(res_mixed[0].score)

    @pytest.mark.asyncio
    async def test_punctuation_normalization(self) -> None:
        """Punctuation differences do not prevent term matching."""
        index = BM25LexicalIndex()
        chunk = _create_chunk("Hello, world! Welcome to RAGForge; ready for search?")
        await index.index_chunks([chunk])

        results = await index.search("world ragforge search", top_k=5)
        assert len(results) == 1
        assert results[0].chunk.id == chunk.id

    @pytest.mark.asyncio
    async def test_unknown_query_terms(self) -> None:
        """Unknown terms do not match; queries with mixed terms score only known terms."""
        index = BM25LexicalIndex()
        chunk = _create_chunk("Vector database indexing with embeddings.")
        await index.index_chunks([chunk])

        # All unknown
        assert await index.search("astronaut spaceship", top_k=5) == []

        # Partially unknown
        results = await index.search("database astronaut", top_k=5)
        assert len(results) == 1
        assert results[0].chunk.id == chunk.id

    @pytest.mark.asyncio
    async def test_ranking_by_bm25_score(self) -> None:
        """Documents with higher term frequency rank ahead of documents with lower frequency."""
        index = BM25LexicalIndex()
        c1 = _create_chunk("hybrid hybrid hybrid retrieval approach")
        c2 = _create_chunk("hybrid search technique with vector store")
        await index.index_chunks([c1, c2])

        results = await index.search("hybrid", top_k=5)
        assert len(results) == 2
        assert results[0].chunk.id == c1.id
        assert results[1].chunk.id == c2.id
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_document_length_normalization(self) -> None:
        """Equal term frequency in shorter documents yields higher BM25 score than in long ones."""
        index = BM25LexicalIndex(k1=1.5, b=0.75)
        # Short document with 1 occurrence of "quantum"
        short_chunk = _create_chunk("quantum computing breakthroughs")
        # Long document with 1 occurrence of "quantum" padded with filler words
        filler = " ".join(["algorithm"] * 60)
        long_chunk = _create_chunk(f"quantum {filler}")

        await index.index_chunks([short_chunk, long_chunk])

        results = await index.search("quantum", top_k=5)
        assert len(results) == 2
        assert results[0].chunk.id == short_chunk.id
        assert results[1].chunk.id == long_chunk.id
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_term_frequency_saturation(self) -> None:
        """BM25 term frequency saturation: marginal gain decreases as tf increases."""
        index = BM25LexicalIndex(k1=1.5, b=0.75)

        # Baseline chunks of equal length to isolate tf effect
        c1 = _create_chunk("term filler1 filler2 filler3")
        c2 = _create_chunk("term term filler2 filler3")
        c9 = _create_chunk("term term term term term term term term term filler3")
        c10 = _create_chunk("term term term term term term term term term term")

        await index.index_chunks([c1, c2, c9, c10])
        res = {r.chunk.id: r.score for r in await index.search("term", top_k=10)}

        # Marginal gain from tf=1 to tf=2 should exceed marginal gain from tf=9 to tf=10
        gain_1_to_2 = res[c2.id] - res[c1.id]
        gain_9_to_10 = res[c10.id] - res[c9.id]
        assert gain_1_to_2 > gain_9_to_10 > 0

    @pytest.mark.asyncio
    async def test_repeated_chunk_update(self) -> None:
        """Re-indexing a chunk replaces postings and length without duplication."""
        index = BM25LexicalIndex()
        shared_id = uuid4()
        c_v1 = _create_chunk("initial ruby on rails framework", chunk_id=shared_id)
        await index.index_chunks([c_v1])

        assert index.count() == 1
        assert len(await index.search("ruby", top_k=5)) == 1
        assert len(await index.search("golang", top_k=5)) == 0

        # Update chunk with new content containing "golang" and removing "ruby"
        c_v2 = _create_chunk("modern golang microservices architecture", chunk_id=shared_id)
        await index.index_chunks([c_v2])

        assert index.count() == 1
        # Old term "ruby" must no longer match
        assert len(await index.search("ruby", top_k=5)) == 0
        # New term "golang" matches
        results = await index.search("golang", top_k=5)
        assert len(results) == 1
        assert results[0].chunk.id == shared_id

    @pytest.mark.asyncio
    async def test_delete_by_chunk_id(self) -> None:
        """Deleting a chunk removes its postings and returns True; nonexistent returns False."""
        index = BM25LexicalIndex()
        c1 = _create_chunk("alpha beta gamma")
        c2 = _create_chunk("delta epsilon zeta")
        await index.index_chunks([c1, c2])

        assert index.count() == 2
        deleted = await index.delete_by_chunk_id(c1.id)
        assert deleted is True
        assert index.count() == 1
        assert await index.get(c1.id) is None

        # Searching for terms from c1 returns nothing
        assert await index.search("alpha", top_k=5) == []
        # c2 remains intact
        assert len(await index.search("delta", top_k=5)) == 1

        # Idempotent deletion of non-existent chunk returns False
        assert await index.delete_by_chunk_id(c1.id) is False
        assert await index.delete_by_chunk_id(uuid4()) is False

    @pytest.mark.asyncio
    async def test_delete_by_document_id(self) -> None:
        """Deleting by document ID purges all chunks for that document and preserves others."""
        index = BM25LexicalIndex()
        target_doc_id = uuid4()
        other_doc_id = uuid4()

        c1 = _create_chunk("distributed storage system part one", doc_id=target_doc_id)
        c2 = _create_chunk("distributed storage system part two", doc_id=target_doc_id)
        c3 = _create_chunk("unrelated relational database notes", doc_id=other_doc_id)

        await index.index_chunks([c1, c2, c3])
        assert index.count() == 3

        await index.delete_by_document_id(target_doc_id)
        assert index.count() == 1

        # Target document chunks no longer match
        assert await index.search("storage", top_k=5) == []
        # Other document chunk still matches
        results = await index.search("relational", top_k=5)
        assert len(results) == 1
        assert results[0].chunk.id == c3.id

        # Idempotent re-deletion
        await index.delete_by_document_id(target_doc_id)
        assert index.count() == 1

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        """Clearing the index resets all storage, lengths, and inverted index state."""
        index = BM25LexicalIndex()
        chunks = [_create_chunk(f"chunk number {i}") for i in range(5)]
        await index.index_chunks(chunks)

        assert index.count() == 5
        await index.clear()

        assert index.count() == 0
        assert len(index) == 0
        assert await index.search("chunk", top_k=5) == []

    @pytest.mark.asyncio
    async def test_deterministic_ordering_tie_breaking(self) -> None:
        """Chunks with identical scores are ordered deterministically by string chunk ID."""
        index = BM25LexicalIndex()
        # Create identical chunks with known sorted IDs
        id_a = UUID("00000000-0000-0000-0000-000000000001")
        id_b = UUID("00000000-0000-0000-0000-000000000002")
        id_c = UUID("00000000-0000-0000-0000-000000000003")

        c_b = _create_chunk("deterministic token stream", chunk_id=id_b)
        c_a = _create_chunk("deterministic token stream", chunk_id=id_a)
        c_c = _create_chunk("deterministic token stream", chunk_id=id_c)

        # Index in non-sorted order
        await index.index_chunks([c_b, c_c, c_a])

        results = await index.search("deterministic", top_k=10)
        assert len(results) == 3
        # Scores are all identical
        assert results[0].score == results[1].score == results[2].score
        # Tie-break order must be id_a, id_b, id_c
        assert [r.chunk.id for r in results] == [id_a, id_b, id_c]

    @pytest.mark.asyncio
    async def test_metadata_filtering(self) -> None:
        """Metadata filters restrict candidate results matching InMemoryVectorStore semantics."""
        index = BM25LexicalIndex()
        doc_id = uuid4()
        c1 = _create_chunk(
            "machine learning pipeline",
            doc_id=doc_id,
            extra={"category": "ai", "status": "active"},
        )
        c2 = _create_chunk(
            "machine learning framework",
            doc_id=doc_id,
            extra={"category": "ai", "status": "deprecated"},
        )
        c3 = _create_chunk(
            "machine learning platform",
            doc_id=uuid4(),
            extra={"category": "infra", "status": "active"},
        )

        await index.index_chunks([c1, c2, c3])

        # Filter by document_id
        res_doc = await index.search("learning", filters={"document_id": doc_id})
        assert len(res_doc) == 2
        assert {r.chunk.id for r in res_doc} == {c1.id, c2.id}

        # Filter by extra metadata key
        res_active_ai = await index.search(
            "learning",
            filters={"category": "ai", "status": "active"},
        )
        assert len(res_active_ai) == 1
        assert res_active_ai[0].chunk.id == c1.id

    @pytest.mark.asyncio
    async def test_top_k_behavior(self) -> None:
        """top_k restricts the number of results, and invalid top_k raises ValueError."""
        index = BM25LexicalIndex()
        chunks = [_create_chunk(f"common keyword document index {i}") for i in range(10)]
        await index.index_chunks(chunks)

        results_3 = await index.search("keyword", top_k=3)
        assert len(results_3) == 3
        assert [r.rank for r in results_3] == [1, 2, 3]

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            await index.search("keyword", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            await index.search("keyword", top_k=-2)

    @pytest.mark.asyncio
    async def test_empty_query(self) -> None:
        """Empty or whitespace-only queries return an empty list without error."""
        index = BM25LexicalIndex()
        await index.index_chunks([_create_chunk("Valid indexed content")])

        assert await index.search("") == []
        assert await index.search("   ") == []
        assert await index.search("!@#$%^&*()") == []

    @pytest.mark.asyncio
    async def test_duplicate_query_terms(self) -> None:
        """Queries containing repeated terms produce consistent scores without duplication error."""
        index = BM25LexicalIndex()
        chunk = _create_chunk("search query processing")
        await index.index_chunks([chunk])

        res_single = await index.search("search", top_k=5)
        res_dup = await index.search("search search search", top_k=5)

        assert len(res_single) == 1
        assert len(res_dup) == 1
        assert res_single[0].score == pytest.approx(res_dup[0].score)

    @pytest.mark.asyncio
    async def test_no_stale_postings_after_deletion_and_update(self) -> None:
        """Internal inverted index removes terms completely when last posting is removed."""
        index = BM25LexicalIndex()
        c1 = _create_chunk("unique_keyword_a and common_term")
        c2 = _create_chunk("unique_keyword_b and common_term")
        await index.index_chunks([c1, c2])

        assert "unique_keyword_a" in index._inverted_index
        assert "unique_keyword_b" in index._inverted_index
        assert "common_term" in index._inverted_index

        # Delete c1 -> unique_keyword_a must be completely removed from _inverted_index
        await index.delete_by_chunk_id(c1.id)
        assert "unique_keyword_a" not in index._inverted_index
        assert "common_term" in index._inverted_index
        assert len(index._inverted_index["common_term"]) == 1

        # Update c2 to remove common_term
        c2_updated = _create_chunk("completely new terms", chunk_id=c2.id)
        await index.index_chunks([c2_updated])
        assert "common_term" not in index._inverted_index
        assert "unique_keyword_b" not in index._inverted_index
        assert "new" in index._inverted_index


def test_tokenize_helper() -> None:
    """Verify tokenize produces normalized alphanumeric tokens."""
    tokens = tokenize("  Hello, World! 123-abc_def...  ")
    assert tokens == ["hello", "world", "123", "abc_def"]
    assert tokenize("") == []
