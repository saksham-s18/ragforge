from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from ragforge.adapters.chunkers import DeterministicChunker
from ragforge.adapters.embeddings import (
    DeterministicEmbeddingProvider,
    FastEmbedProvider,
)
from ragforge.adapters.state import InMemoryIndexStateStore
from ragforge.adapters.vector_stores.qdrant import QdrantVectorStore
from ragforge.services.indexing import IndexingService
from ragforge.services.retrieval import RetrievalService

QDRANT_TEST_URL = "http://localhost:6333"


@pytest.fixture
async def qdrant_client() -> AsyncQdrantClient:
    """Fixture providing a connected AsyncQdrantClient for integration tests."""
    client = AsyncQdrantClient(url=QDRANT_TEST_URL, check_compatibility=False)
    yield client
    await client.close()


@pytest.fixture
async def temp_collection(qdrant_client: AsyncQdrantClient) -> str:
    """Fixture ensuring an ephemeral collection is cleaned up after testing."""
    collection_name = f"test_idx_col_{uuid4().hex[:8]}"
    yield collection_name
    try:
        if await qdrant_client.collection_exists(collection_name):
            await qdrant_client.delete_collection(collection_name)
    except Exception:
        pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_indexing_pipeline_new_document_and_retrieval(
    qdrant_client: AsyncQdrantClient, temp_collection: str, tmp_path: Path
) -> None:
    """Verify new documents are indexed to real Qdrant and searchable via RetrievalService."""
    dim = 16
    provider = DeterministicEmbeddingProvider(dimension=dim)
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=dim,
    )
    state = InMemoryIndexStateStore()
    chunker = DeterministicChunker(chunk_size=20, chunk_overlap=5)
    service = IndexingService(
        embedding_provider=provider,
        vector_store=store,
        chunker=chunker,
        state_store=state,
        batch_size=4,
    )

    doc_file = tmp_path / "knowledge.md"
    doc_file.write_text(
        "# RAG Architecture\n"
        "Retrieval-Augmented Generation combines dense search with generative language models.\n\n"
        "## Vector Database\n"
        "Qdrant stores dense vector embeddings and provides high-speed similarity search.",
        encoding="utf-8",
    )

    result = await service.index_path(doc_file)

    assert result.discovered_documents == 1
    assert result.indexed_documents == 1
    assert result.updated_documents == 0
    assert result.failed_documents == 0
    assert result.chunks_created >= 2
    assert result.vectors_upserted == result.chunks_created
    assert await store.count() == result.chunks_created

    # Verify retrieval against indexed chunks in Qdrant
    retrieval_service = RetrievalService(embedding_provider=provider, vector_store=store)
    retrieved = await retrieval_service.retrieve("Qdrant stores dense vector embeddings", top_k=2)

    assert len(retrieved) > 0
    assert "dense vector" in retrieved[0].chunk.content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_indexing_pipeline_incremental_update_removes_stale_chunks(
    qdrant_client: AsyncQdrantClient, temp_collection: str, tmp_path: Path
) -> None:
    """Verify modifying a document removes stale chunks from real Qdrant and updates state."""
    dim = 16
    provider = DeterministicEmbeddingProvider(dimension=dim)
    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=dim,
    )
    state = InMemoryIndexStateStore()
    chunker = DeterministicChunker(chunk_size=60, chunk_overlap=10)
    service = IndexingService(
        embedding_provider=provider,
        vector_store=store,
        chunker=chunker,
        state_store=state,
        batch_size=4,
    )

    doc_file = tmp_path / "mutable_doc.txt"
    # Create long text producing 3 or more chunks
    paragraph = "Sentence demonstrating production indexing pipeline with Qdrant. " * 8
    doc_file.write_text(
        f"Section A:\n{paragraph}\n\nSection B:\n{paragraph}\n\nSection C:\n{paragraph}",
        encoding="utf-8",
    )

    # Initial indexing run
    res1 = await service.index_path(doc_file)
    assert res1.indexed_documents == 1
    old_chunks = res1.chunks_created
    assert old_chunks >= 3
    assert await store.count() == old_chunks

    # Second run without changes should skip
    res2 = await service.index_path(doc_file)
    assert res2.skipped_documents == 1
    assert res2.updated_documents == 0
    assert res2.vectors_upserted == 0
    assert await store.count() == old_chunks

    # Modify document to have only a single short sentence
    doc_file.write_text("Only a single short updated sentence.", encoding="utf-8")

    # Third run should update and purge stale chunks from Qdrant
    res3 = await service.index_path(doc_file)
    assert res3.updated_documents == 1
    assert res3.chunks_created == 1
    assert res3.vectors_upserted == 1

    # Stale chunks from version 1 must not remain in Qdrant
    assert await store.count() == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_indexing_pipeline_fastembed_production_integration(
    qdrant_client: AsyncQdrantClient, temp_collection: str, tmp_path: Path
) -> None:
    """Verify FastEmbed provider integrates end-to-end with IndexingService and Qdrant."""
    provider = FastEmbedProvider(model_name="BAAI/bge-small-en-v1.5", batch_size=4)
    assert provider.dimension == 384

    store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=temp_collection,
        dimension=384,
    )
    state = InMemoryIndexStateStore()
    service = IndexingService(
        embedding_provider=provider,
        vector_store=store,
        state_store=state,
        batch_size=4,
    )

    doc1 = tmp_path / "deep_learning.txt"
    doc1.write_text(
        "Transformers and neural networks form the basis of modern language processing.",
        encoding="utf-8",
    )

    doc2 = tmp_path / "gardening.txt"
    doc2.write_text(
        "Tomatoes require well-draining soil and direct sunlight during the growing season.",
        encoding="utf-8",
    )

    result = await service.index_path(tmp_path)
    assert result.discovered_documents == 2
    assert result.indexed_documents == 2
    assert await store.count() >= 2

    # Query using RetrievalService with FastEmbed
    retrieval = RetrievalService(embedding_provider=provider, vector_store=store)
    ai_results = await retrieval.retrieve("neural network transformers NLP", top_k=2)

    assert len(ai_results) >= 1
    assert "Transformers and neural networks" in ai_results[0].chunk.content
