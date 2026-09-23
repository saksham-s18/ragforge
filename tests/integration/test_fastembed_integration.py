from uuid import uuid4

import pytest

from ragforge.adapters.embeddings.fastembed import FastEmbedProvider
from ragforge.adapters.vector_stores.in_memory import InMemoryVectorStore
from ragforge.core.similarity import cosine_similarity
from ragforge.domain.models import Chunk, ChunkMetadata
from ragforge.services.retrieval import RetrievalService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fastembed_real_model_loading_and_dimension() -> None:
    """Verify real FastEmbed model loads and produces 384-dimensional embeddings."""
    provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")
    assert provider.dimension == 384

    query_vec = await provider.embed_query("Semantic search test with real ONNX model.")
    assert len(query_vec) == 384
    assert all(isinstance(x, float) for x in query_vec)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fastembed_real_semantic_discrimination() -> None:
    """Verify real embeddings accurately capture semantic similarity and separation."""
    provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")

    query = "artificial intelligence and machine learning"
    passage_ml = "Deep learning and neural network architectures represent modern AI methodologies."
    passage_cooking = "To bake sourdough bread, preheat the Dutch oven to 450 degrees Fahrenheit."

    query_vec = await provider.embed_query(query)
    passage_vectors = await provider.embed_texts([passage_ml, passage_cooking])

    sim_ml = cosine_similarity(query_vec, passage_vectors[0])
    sim_cooking = cosine_similarity(query_vec, passage_vectors[1])

    # AI passage must have significantly higher similarity than baking passage
    assert sim_ml > sim_cooking
    assert sim_ml > 0.6
    assert (sim_ml - sim_cooking) > 0.2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fastembed_real_retrieval_service_flow() -> None:
    """Verify end-to-end retrieval using real FastEmbed embeddings in RetrievalService."""
    provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5")
    store = InMemoryVectorStore(dimension=384)
    service = RetrievalService(embedding_provider=provider, vector_store=store)

    doc_id = uuid4()
    chunk_python = Chunk(
        document_id=doc_id,
        chunk_index=0,
        content="Python is an interpreted programming language often used in data engineering.",
        token_count=12,
        content_hash="hash_python",
        metadata=ChunkMetadata(section_header="Programming Languages"),
    )
    chunk_astronomy = Chunk(
        document_id=doc_id,
        chunk_index=1,
        content="Hubble and James Webb space telescopes observe distant redshifted galaxies.",
        token_count=11,
        content_hash="hash_astronomy",
        metadata=ChunkMetadata(section_header="Astrophysics"),
    )

    await service.index_chunks([chunk_python, chunk_astronomy])

    # Search for programming language query
    results = await service.retrieve("coding in python", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == chunk_python.id
    assert results[0].score > results[1].score

    # Search for astrophysics query
    astro_results = await service.retrieve("space telescopes and astronomy", top_k=2)
    assert len(astro_results) == 2
    assert astro_results[0].chunk.id == chunk_astronomy.id
    assert astro_results[0].score > astro_results[1].score
