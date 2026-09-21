import math

import pytest

from ragforge.adapters.embeddings import DeterministicEmbeddingProvider
from ragforge.core.similarity import cosine_similarity


@pytest.mark.asyncio
async def test_embedding_deterministic_output() -> None:
    """Verify repeated embeddings of identical text produce exactly equal vectors."""
    provider = DeterministicEmbeddingProvider(dimension=64)
    text = "RAGForge deterministic embedding provider verification."

    v1 = await provider.embed_query(text)
    v2 = await provider.embed_query(text)

    assert v1 == v2
    assert len(v1) == 64


@pytest.mark.asyncio
async def test_embedding_same_text_produces_same_vector() -> None:
    """Verify separate instances of provider produce identical vectors for same text."""
    p1 = DeterministicEmbeddingProvider(dimension=32)
    p2 = DeterministicEmbeddingProvider(dimension=32)
    text = "Reproducible feature hashing test."

    v1 = await p1.embed_query(text)
    v2 = await p2.embed_query(text)

    assert v1 == v2


@pytest.mark.asyncio
async def test_embedding_different_text_produces_distinguishable_vectors() -> None:
    """Verify different texts produce distinct vector representations."""
    provider = DeterministicEmbeddingProvider(dimension=64)
    v_fastapi = await provider.embed_query("FastAPI REST endpoints and routing")
    v_astronomy = await provider.embed_query("Telescopes, stars, and galaxies in outer space")

    assert v_fastapi != v_astronomy
    sim = cosine_similarity(v_fastapi, v_astronomy)
    assert sim < 0.95


@pytest.mark.asyncio
async def test_embedding_fixed_dimension() -> None:
    """Verify provider strictly adheres to configured vector dimension."""
    p64 = DeterministicEmbeddingProvider(dimension=64)
    p128 = DeterministicEmbeddingProvider(dimension=128)

    v64 = await p64.embed_query("Check dimension 64")
    v128 = await p128.embed_query("Check dimension 128")

    assert len(v64) == 64
    assert len(v128) == 128


def test_embedding_invalid_dimension() -> None:
    """Verify invalid dimensions raise ValueError."""
    with pytest.raises(ValueError, match="Embedding dimension must be greater than 0"):
        DeterministicEmbeddingProvider(dimension=0)

    with pytest.raises(ValueError, match="Embedding dimension must be greater than 0"):
        DeterministicEmbeddingProvider(dimension=-16)


@pytest.mark.asyncio
async def test_embedding_batch_embedding() -> None:
    """Verify embed_texts correctly processes a batch of multiple strings."""
    provider = DeterministicEmbeddingProvider(dimension=32)
    texts = ["Document one", "Document two", "Document three"]

    vectors = await provider.embed_texts(texts)

    assert len(vectors) == 3
    for vec in vectors:
        assert len(vec) == 32


@pytest.mark.asyncio
async def test_embedding_normalization_behavior() -> None:
    """Verify non-empty text vectors are unit-length normalized (L2 norm = 1.0)."""
    provider = DeterministicEmbeddingProvider(dimension=64)
    vec = await provider.embed_query("Text normalization verification.")

    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, rel=1e-5)


@pytest.mark.asyncio
async def test_embedding_empty_and_whitespace_text() -> None:
    """Verify empty or whitespace-only strings return zero vectors of exact dimension."""
    provider = DeterministicEmbeddingProvider(dimension=32)

    v_empty = await provider.embed_query("")
    v_spaces = await provider.embed_query("   \t  \n  ")

    assert len(v_empty) == 32
    assert all(x == 0.0 for x in v_empty)
    assert len(v_spaces) == 32
    assert all(x == 0.0 for x in v_spaces)


@pytest.mark.asyncio
async def test_embedding_query_matches_batch_element() -> None:
    """Verify embed_query output matches embed_texts for the same string."""
    provider = DeterministicEmbeddingProvider(dimension=64)
    text = "Consistency check between query and text methods."

    query_vec = await provider.embed_query(text)
    texts_vec = (await provider.embed_texts([text]))[0]

    assert query_vec == texts_vec
