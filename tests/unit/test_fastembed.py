from collections.abc import Iterator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from ragforge.adapters.embeddings.fastembed import FastEmbedProvider
from ragforge.adapters.vector_stores.in_memory import InMemoryVectorStore
from ragforge.domain.exceptions import (
    EmbeddingError,
    EmbeddingModelNotFoundError,
    VectorDimensionMismatchError,
)
from ragforge.domain.models import Chunk, ChunkMetadata
from ragforge.ports.embeddings import BaseEmbeddingProvider
from ragforge.services.retrieval import RetrievalService


class DummyMockModel:
    """Mock FastEmbed TextEmbedding for fast, offline unit testing."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def passage_embed(self, texts: list[str], batch_size: int = 32) -> Iterator[np.ndarray]:
        for text in texts:
            # Deterministic float array based on input length
            base_val = float((len(text) * 7) % 100) / 100.0
            vec = np.linspace(base_val, base_val + 1.0, num=self.dim, dtype=np.float32)
            yield vec

    def query_embed(self, query: str) -> Iterator[np.ndarray]:
        base_val = float((len(query) * 7) % 100) / 100.0
        vec = np.linspace(base_val, base_val + 1.0, num=self.dim, dtype=np.float32)
        yield vec


def test_fastembed_implements_port() -> None:
    """Verify FastEmbedProvider is an instance of BaseEmbeddingProvider."""
    mock_model = DummyMockModel()
    provider = FastEmbedProvider(model=mock_model, dimension=384)
    assert isinstance(provider, BaseEmbeddingProvider)


def test_fastembed_initialization_defaults() -> None:
    """Verify default initialization parameters."""
    mock_model = DummyMockModel()
    provider = FastEmbedProvider(model=mock_model)
    assert provider.model_name == "BAAI/bge-small-en-v1.5"
    assert provider.batch_size == 32
    assert provider.dimension == 384
    assert provider.cache_dir is None
    assert provider.threads is None


def test_fastembed_custom_parameters() -> None:
    """Verify custom initialization options."""
    mock_model = DummyMockModel(dim=768)
    provider = FastEmbedProvider(
        model_name="BAAI/bge-base-en-v1.5",
        cache_dir="/custom/cache",
        threads=4,
        batch_size=16,
        dimension=768,
        model=mock_model,
    )
    assert provider.model_name == "BAAI/bge-base-en-v1.5"
    assert provider.batch_size == 16
    assert provider.dimension == 768
    assert provider.threads == 4
    assert provider.cache_dir is not None
    assert str(provider.cache_dir).replace("\\", "/").endswith("custom/cache")


def test_fastembed_invalid_parameters() -> None:
    """Verify invalid batch_size or dimension raise ValueError."""
    mock_model = DummyMockModel()
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        FastEmbedProvider(batch_size=0, model=mock_model)

    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        FastEmbedProvider(batch_size=-10, model=mock_model)

    with pytest.raises(ValueError, match="dimension must be greater than 0"):
        FastEmbedProvider(dimension=0, model=mock_model)

    with pytest.raises(ValueError, match="dimension must be greater than 0"):
        FastEmbedProvider(dimension=-64, model=mock_model)


def test_fastembed_initialization_model_error_translation() -> None:
    """Verify initialization failures are cleanly translated into domain exceptions."""
    with patch(
        "ragforge.adapters.embeddings.fastembed.TextEmbedding",
        side_effect=ValueError("Model not supported"),
    ):
        with pytest.raises(EmbeddingModelNotFoundError, match="not supported or not found"):
            FastEmbedProvider(model_name="nonexistent/model")

    with patch(
        "ragforge.adapters.embeddings.fastembed.TextEmbedding",
        side_effect=RuntimeError("Download failed"),
    ):
        with pytest.raises(EmbeddingError, match="Failed to initialize FastEmbed model"):
            FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")


@pytest.mark.asyncio
async def test_fastembed_embed_texts_batch() -> None:
    """Verify batch embedding produces correct count, dimensions, and types."""
    mock_model = DummyMockModel(dim=384)
    provider = FastEmbedProvider(model=mock_model, dimension=384)

    texts = [
        "First document paragraph.",
        "Second document paragraph with more detailed content.",
        "Third document chunk.",
    ]

    embeddings = await provider.embed_texts(texts)

    assert len(embeddings) == 3
    for vec in embeddings:
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)


@pytest.mark.asyncio
async def test_fastembed_embed_texts_empty() -> None:
    """Verify empty text list returns an empty list immediately."""
    mock_model = DummyMockModel(dim=384)
    provider = FastEmbedProvider(model=mock_model, dimension=384)

    assert await provider.embed_texts([]) == []


@pytest.mark.asyncio
async def test_fastembed_embed_query_valid() -> None:
    """Verify embedding a single query string produces expected dimension and type."""
    mock_model = DummyMockModel(dim=384)
    provider = FastEmbedProvider(model=mock_model, dimension=384)

    query = "What is agentic retrieval augmented generation?"
    vec = await provider.embed_query(query)

    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


@pytest.mark.asyncio
async def test_fastembed_embed_query_empty_and_whitespace() -> None:
    """Verify empty or whitespace query returns zero vector of exact dimension."""
    mock_model = DummyMockModel(dim=384)
    provider = FastEmbedProvider(model=mock_model, dimension=384)

    v_empty = await provider.embed_query("")
    v_spaces = await provider.embed_query("    \t \n  ")

    assert len(v_empty) == 384
    assert all(x == 0.0 for x in v_empty)
    assert len(v_spaces) == 384
    assert all(x == 0.0 for x in v_spaces)


@pytest.mark.asyncio
async def test_fastembed_deterministic_for_identical_input() -> None:
    """Verify identical queries yield identical vector representations."""
    mock_model = DummyMockModel(dim=384)
    provider = FastEmbedProvider(model=mock_model, dimension=384)

    text = "Deterministic vector consistency check."
    v1 = await provider.embed_query(text)
    v2 = await provider.embed_query(text)

    assert v1 == v2


@pytest.mark.asyncio
async def test_fastembed_error_handling_during_embedding() -> None:
    """Verify runtime embedding generation errors raise EmbeddingError."""
    failing_model = MagicMock()
    failing_model.passage_embed.side_effect = RuntimeError("ONNX execution error")
    failing_model.query_embed.side_effect = RuntimeError("Tokenizer error")

    provider = FastEmbedProvider(model=failing_model, dimension=384)

    with pytest.raises(EmbeddingError, match="Failed to generate text embeddings"):
        await provider.embed_texts(["Valid text"])

    with pytest.raises(EmbeddingError, match="Failed to generate query embedding"):
        await provider.embed_query("Valid query")


@pytest.mark.asyncio
async def test_fastembed_empty_query_embed_output_handling() -> None:
    """Verify query_embed producing empty iterator raises EmbeddingError."""
    empty_generator_model = MagicMock()
    empty_generator_model.query_embed.return_value = iter([])

    provider = FastEmbedProvider(model=empty_generator_model, dimension=384)

    with pytest.raises(EmbeddingError, match="produced no output vector"):
        await provider.embed_query("Non-empty query")


@pytest.mark.asyncio
async def test_fastembed_retrieval_service_integration() -> None:
    """Verify FastEmbedProvider integrates with RetrievalService and InMemoryVectorStore."""
    mock_model = DummyMockModel(dim=384)
    provider = FastEmbedProvider(model=mock_model, dimension=384)
    store = InMemoryVectorStore(dimension=384)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    chunk = Chunk(
        document_id=uuid4(),
        chunk_index=0,
        content="Retrieval service integration with FastEmbed.",
        token_count=10,
        content_hash="hash_integration_test",
        metadata=ChunkMetadata(),
    )

    await service.index_chunks([chunk])
    results = await service.retrieve("FastEmbed integration", top_k=5)

    assert len(results) == 1
    assert results[0].chunk.id == chunk.id
    assert len(chunk.dense_vector or []) == 384


@pytest.mark.asyncio
async def test_fastembed_vector_dimension_mismatch_with_store() -> None:
    """Verify vector dimension mismatch between FastEmbed and vector store raises exception."""
    mock_model = DummyMockModel(dim=384)
    provider = FastEmbedProvider(model=mock_model, dimension=384)
    store_64 = InMemoryVectorStore(dimension=64)
    service = RetrievalService(embedding_provider=provider, vector_store=store_64)

    chunk = Chunk(
        document_id=uuid4(),
        chunk_index=0,
        content="Chunk to trigger dimension mismatch.",
        token_count=8,
        content_hash="hash_mismatch_test",
        metadata=ChunkMetadata(),
    )

    with pytest.raises(VectorDimensionMismatchError, match="dimension 384 does not match"):
        await service.index_chunks([chunk])
