"""Unit tests for Stage 11 Step 4 dependency wiring, shared lexical index, and RRF configuration."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import Request

from ragforge.adapters.fusion import ReciprocalRankFusion
from ragforge.adapters.lexical import BM25LexicalIndex
from ragforge.adapters.rerankers import DeterministicReranker
from ragforge.adapters.vector_stores import QdrantVectorStore
from ragforge.api.app import create_app
from ragforge.api.dependencies import (
    get_embedding_provider,
    get_fusion,
    get_indexing_service,
    get_lexical_index,
    get_primary_llm_provider,
    get_rag_service,
    get_reranker,
    get_retrieval_service,
    get_vector_store,
    reset_dependencies,
)
from ragforge.core.config import Settings
from ragforge.domain.enums import SearchStrategy
from ragforge.domain.models import Chunk, ChunkMetadata
from ragforge.services.llm_router import LLMRouter
from ragforge.services.rag import RAGGenerationService
from ragforge.services.retrieval import RetrievalService
from tests.unit.test_rag_service import MockLLMProvider, make_retrieved_chunk


@pytest.fixture(autouse=True)
def clean_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Ensure clean singletons and isolated lexical index file before and after tests."""
    monkeypatch.setenv("RAGFORGE_LEXICAL_INDEX_FILE", str(tmp_path / "test_lexical_index.json"))
    reset_dependencies()
    yield
    reset_dependencies()


# ---------------------------------------------------------------------------
# Requirement 7 & 15: Hybrid disabled -> dense-only, no lexical index
# ---------------------------------------------------------------------------


def test_dependencies_when_hybrid_disabled() -> None:
    """Verify that when hybrid_enabled=False, no lexical index or fusion adapter is created."""
    settings = Settings(hybrid_enabled=False)

    lexical = get_lexical_index(settings)
    assert lexical is None

    fusion = get_fusion(settings)
    assert fusion is None

    retrieval_service = get_retrieval_service(settings)
    assert retrieval_service._lexical_index is None
    assert retrieval_service._fusion is None

    indexing_service = get_indexing_service(settings)
    assert indexing_service._lexical_index is None


# ---------------------------------------------------------------------------
# Requirement 7, 8 & 9: Hybrid enabled -> BM25 + RRF injected with configured params
# ---------------------------------------------------------------------------


def test_dependencies_when_hybrid_enabled() -> None:
    """Verify that BM25LexicalIndex and ReciprocalRankFusion are created with configured values."""
    settings = Settings(
        hybrid_enabled=True,
        fusion_strategy="rrf",
        rrf_k=42,
        rrf_dense_weight=1.5,
        rrf_lexical_weight=0.8,
        hybrid_candidate_k=25,
    )

    lexical = get_lexical_index(settings)
    assert isinstance(lexical, BM25LexicalIndex)

    fusion = get_fusion(settings)
    assert isinstance(fusion, ReciprocalRankFusion)
    assert fusion.k == 42
    assert fusion.dense_weight == 1.5
    assert fusion.lexical_weight == 0.8

    retrieval_service = get_retrieval_service(settings)
    assert retrieval_service._lexical_index is lexical
    assert isinstance(retrieval_service._fusion, ReciprocalRankFusion)
    assert retrieval_service._fusion.k == 42
    assert retrieval_service._fusion.dense_weight == 1.5
    assert retrieval_service._fusion.lexical_weight == 0.8

    indexing_service = get_indexing_service(settings)
    assert indexing_service._lexical_index is lexical


def test_same_lexical_index_instance_shared_between_indexing_and_retrieval() -> None:
    """Verify that IndexingService and RetrievalService receive the EXACT SAME lexical instance."""
    settings = Settings(hybrid_enabled=True)

    retrieval_service = get_retrieval_service(settings)
    indexing_service = get_indexing_service(settings)

    assert retrieval_service._lexical_index is not None
    assert indexing_service._lexical_index is not None
    assert retrieval_service._lexical_index is indexing_service._lexical_index


@pytest.mark.asyncio
async def test_shared_lexical_index_reflects_indexed_documents() -> None:
    """Verify documents indexed via shared instance are immediately searchable by retrieval."""
    settings = Settings(hybrid_enabled=True)

    retrieval_service = get_retrieval_service(settings)
    indexing_service = get_indexing_service(settings)

    shared_lexical = indexing_service._lexical_index
    assert shared_lexical is not None

    chunk = Chunk(
        id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content="Antigravity agentic coding platform with hybrid lexical search",
        token_count=8,
        content_hash="hash123",
        metadata=ChunkMetadata(),
    )
    await shared_lexical.index_chunks([chunk])

    # Search through retrieval service's lexical index
    retrieval_lexical = retrieval_service._lexical_index
    assert retrieval_lexical is not None
    results = await retrieval_lexical.search("hybrid lexical search", top_k=5)
    assert len(results) == 1
    assert results[0].chunk.id == chunk.id


def test_reset_dependencies_clears_singleton() -> None:
    """Verify reset_dependencies clears the shared singleton instance."""
    settings = Settings(hybrid_enabled=True)

    idx1 = get_lexical_index(settings)
    idx2 = get_lexical_index(settings)
    assert idx1 is idx2

    reset_dependencies()

    idx3 = get_lexical_index(settings)
    assert idx3 is not idx1


def test_unsupported_fusion_strategy_raises_value_error() -> None:
    """Verify get_fusion raises ValueError for unsupported fusion strategy."""
    settings = Settings(hybrid_enabled=True)
    # Manually bypass pydantic validation to verify get_fusion defensive check
    object.__setattr__(settings, "fusion_strategy", "unsupported_fusion")

    with pytest.raises(ValueError) as exc_info:
        get_fusion(settings)
    assert "Unsupported fusion strategy" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Requirement 10: hybrid_candidate_k behavior and Stage 10 reranker / slicing semantics
# ---------------------------------------------------------------------------


def test_rag_service_hybrid_candidate_k_injection() -> None:
    """Verify get_rag_service injects configured hybrid_candidate_k."""
    settings = Settings(
        groq_api_key="gsk_test_mock_key_12345",
        hybrid_candidate_k=35,
        _env_file=None,
    )
    rag_service = get_rag_service(settings)
    assert rag_service._hybrid_candidate_k == 35


@pytest.mark.asyncio
async def test_rag_generation_service_hybrid_candidate_pool_slicing() -> None:
    """Verify that hybrid search retrieves candidate_k items and slices to top_k for LLM context."""
    candidates = [
        make_retrieved_chunk(content=f"Candidate chunk {i}", score=1.0 - (i * 0.04), rank=i + 1)
        for i in range(20)
    ]

    mock_retrieval = AsyncMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = candidates

    router = LLMRouter(primary_provider=MockLLMProvider())
    rag_service = RAGGenerationService(
        retrieval_service=mock_retrieval,
        llm_router=router,
        default_top_k=5,
        hybrid_candidate_k=20,
    )

    response = await rag_service.generate_answer(
        question="What is hybrid retrieval?",
        top_k=5,
        search_strategy=SearchStrategy.HYBRID,
    )

    # Retrieval should be called with max(top_k=5, candidate_k=20) = 20
    mock_retrieval.retrieve.assert_awaited_once_with(
        query="What is hybrid retrieval?",
        top_k=20,
        filters=None,
        score_threshold=None,
        search_strategy=SearchStrategy.HYBRID,
    )

    # Only top_k=5 chunks passed as sources to context and response
    assert len(response.sources) == 5
    assert response.metadata["retrieved_count"] == 20
    assert response.metadata["context_count"] == 5
    assert response.metadata["top_k"] == 5


@pytest.mark.asyncio
async def test_rag_generation_service_hybrid_top_k_larger_than_candidate_k() -> None:
    """Verify that when top_k > hybrid_candidate_k, candidate_k scales up to top_k."""
    candidates = [
        make_retrieved_chunk(content=f"Candidate chunk {i}", score=1.0 - (i * 0.02), rank=i + 1)
        for i in range(25)
    ]

    mock_retrieval = AsyncMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = candidates

    router = LLMRouter(primary_provider=MockLLMProvider())
    rag_service = RAGGenerationService(
        retrieval_service=mock_retrieval,
        llm_router=router,
        default_top_k=5,
        hybrid_candidate_k=15,
    )

    response = await rag_service.generate_answer(
        question="What is hybrid retrieval?",
        top_k=25,
        search_strategy=SearchStrategy.HYBRID,
    )

    # Candidate k should be max(25, 15) = 25
    mock_retrieval.retrieve.assert_awaited_once_with(
        query="What is hybrid retrieval?",
        top_k=25,
        filters=None,
        score_threshold=None,
        search_strategy=SearchStrategy.HYBRID,
    )

    assert len(response.sources) == 25
    assert response.metadata["retrieved_count"] == 25
    assert response.metadata["context_count"] == 25


@pytest.mark.asyncio
async def test_rag_generation_service_dense_ignores_hybrid_candidate_k() -> None:
    """Verify dense and omitted strategies retrieve exactly top_k chunks without inflating."""
    mock_retrieval = AsyncMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = [make_retrieved_chunk()]

    router = LLMRouter(primary_provider=MockLLMProvider())
    rag_service = RAGGenerationService(
        retrieval_service=mock_retrieval,
        llm_router=router,
        default_top_k=5,
        hybrid_candidate_k=50,
    )

    # 1. Explicit DENSE
    await rag_service.generate_answer(
        question="Test dense",
        top_k=3,
        search_strategy=SearchStrategy.DENSE,
    )
    mock_retrieval.retrieve.assert_awaited_with(
        query="Test dense",
        top_k=3,
        filters=None,
        score_threshold=None,
        search_strategy=SearchStrategy.DENSE,
    )

    # 2. Omitted strategy
    mock_retrieval.reset_mock()
    await rag_service.generate_answer(
        question="Test omitted",
        top_k=4,
        search_strategy=None,
    )
    mock_retrieval.retrieve.assert_awaited_with(
        query="Test omitted",
        top_k=4,
        filters=None,
        score_threshold=None,
    )


@pytest.mark.asyncio
async def test_hybrid_and_reranking_wiring_end_to_end() -> None:
    """Verify that when both hybrid_enabled and reranking_enabled are True:
    1. get_rag_service injects DeterministicReranker and hybrid_candidate_k=20.
    2. Hybrid retrieval fetches 20 candidates into the reranker.
    3. The final response contains exactly 5 results/context items.
    """
    settings = Settings(
        groq_api_key="gsk_test_mock_key_12345",
        hybrid_enabled=True,
        reranking_enabled=True,
        reranker_provider="deterministic",
        hybrid_candidate_k=20,
        rerank_top_k=5,
        _env_file=None,
    )

    rag_service = get_rag_service(settings)
    assert rag_service._reranker is not None
    assert isinstance(rag_service._reranker, DeterministicReranker)
    assert rag_service._reranking_enabled is True
    assert rag_service._hybrid_candidate_k == 20

    candidates = [
        make_retrieved_chunk(
            content=f"Candidate document {i} with specific keywords like neural search.",
            score=0.035 - (i * 0.001),
            rank=i + 1,
            section=f"Section {i}",
            source_path=f"/data/doc_{i}.md",
        )
        for i in range(20)
    ]
    for idx, c in enumerate(candidates):
        c.dense_score = 0.90 - idx * 0.02
        c.lexical_score = 0.45 + idx * 0.01
        c.retrieval_type = "hybrid"

    # Replace retrieval service and llm router with mocks for this test
    mock_retrieval = AsyncMock(spec=RetrievalService)
    mock_retrieval.retrieve.return_value = candidates
    rag_service._retrieval_service = mock_retrieval

    llm_provider = MockLLMProvider(answer_text="Hybrid search summary.")
    rag_service._llm_router = LLMRouter(primary_provider=llm_provider)

    response = await rag_service.generate_answer(
        question="Tell me about neural search",
        top_k=5,
        search_strategy=SearchStrategy.HYBRID,
    )

    # 1. Retrieval called with candidate pool of 20
    mock_retrieval.retrieve.assert_awaited_once_with(
        query="Tell me about neural search",
        top_k=20,
        filters=None,
        score_threshold=None,
        search_strategy=SearchStrategy.HYBRID,
    )

    # 2. Final response sources sliced to top 5
    assert len(response.sources) == 5
    assert response.metadata["retrieved_count"] == 20
    assert response.metadata["candidate_count"] == 20
    assert response.metadata["context_count"] == 5
    assert response.metadata["top_k"] == 5
    assert response.metadata["reranked"] is True

    # 3. Provenance and original scores strictly retained on source references
    for src in response.sources:
        assert src.score > 0
        assert src.rerank_score is not None
        assert src.source_path is not None
        assert src.document_title is not None


# ---------------------------------------------------------------------------
# Issue 1: Adapter Reuse, App Scoping, and Qdrant Lifespan Shutdown
# ---------------------------------------------------------------------------


def test_request_independent_adapters_reused_per_app() -> None:
    """Verify adapters and services are cached on app.state and reused across requests."""
    settings = Settings(
        groq_api_key="gsk_test_mock_key_12345",
        hybrid_enabled=True,
        reranking_enabled=True,
        _env_file=None,
    )
    app = create_app(settings=settings)

    req1 = Request(scope={"type": "http", "app": app})
    req2 = Request(scope={"type": "http", "app": app})

    # Vector store
    vs1 = get_vector_store(request=req1)
    vs2 = get_vector_store(request=req2)
    assert vs1 is vs2

    # Embedding provider
    ep1 = get_embedding_provider(request=req1)
    ep2 = get_embedding_provider(request=req2)
    assert ep1 is ep2

    # Primary LLM provider
    llm1 = get_primary_llm_provider(request=req1)
    llm2 = get_primary_llm_provider(request=req2)
    assert llm1 is llm2

    # Lexical index
    lex1 = get_lexical_index(request=req1)
    lex2 = get_lexical_index(request=req2)
    assert lex1 is not None
    assert lex1 is lex2

    # Fusion
    fus1 = get_fusion(request=req1)
    fus2 = get_fusion(request=req2)
    assert fus1 is not None
    assert fus1 is fus2

    # Reranker
    rerank1 = get_reranker(request=req1)
    rerank2 = get_reranker(request=req2)
    assert rerank1 is not None
    assert rerank1 is rerank2

    # Composite services
    ret1 = get_retrieval_service(request=req1)
    ret2 = get_retrieval_service(request=req2)
    assert ret1 is ret2

    idx1 = get_indexing_service(request=req1)
    idx2 = get_indexing_service(request=req2)
    assert idx1 is idx2

    rag1 = get_rag_service(request=req1)
    rag2 = get_rag_service(request=req2)
    assert rag1 is rag2


def test_app_isolation_no_leakage() -> None:
    """Verify different application instances have isolated state and no leakage."""
    settings1 = Settings(
        qdrant_collection="collection_alpha",
        groq_api_key="gsk_key_alpha",
        _env_file=None,
    )
    settings2 = Settings(
        qdrant_collection="collection_beta",
        groq_api_key="gsk_key_beta",
        _env_file=None,
    )
    app1 = create_app(settings=settings1)
    app2 = create_app(settings=settings2)

    req1 = Request(scope={"type": "http", "app": app1})
    req2 = Request(scope={"type": "http", "app": app2})

    store1 = get_vector_store(request=req1)
    store2 = get_vector_store(request=req2)

    assert store1 is not store2
    assert isinstance(store1, QdrantVectorStore)
    assert isinstance(store2, QdrantVectorStore)
    assert store1.collection_name == "collection_alpha"
    assert store2.collection_name == "collection_beta"


@pytest.mark.asyncio
async def test_lifespan_closes_owned_vector_store() -> None:
    """Verify application lifespan shutdown closes owned vector store client."""
    settings = Settings(
        groq_api_key="gsk_test_mock_key_12345",
        _env_file=None,
    )
    app = create_app(settings=settings)

    async with app.router.lifespan_context(app):
        req = Request(scope={"type": "http", "app": app})
        store = get_vector_store(request=req)
        assert isinstance(store, QdrantVectorStore)
        mock_client = AsyncMock()
        store._client = mock_client
        store._owned_client = True

    # After exiting lifespan context, store._client.close must have been awaited
    mock_client.close.assert_awaited_once()


def test_dependency_overrides_respected_by_composite_services() -> None:
    """Verify FastAPI dependency_overrides are honored by downstream service factories."""
    settings = Settings(
        groq_api_key="gsk_test_mock_key_12345",
        _env_file=None,
    )
    app = create_app(settings=settings)
    mock_store = MagicMock()
    app.dependency_overrides[get_vector_store] = lambda: mock_store

    req = Request(scope={"type": "http", "app": app})
    assert get_vector_store(request=req) is mock_store

    retrieval = get_retrieval_service(request=req)
    assert retrieval._vector_store is mock_store

    indexing = get_indexing_service(request=req)
    assert indexing._vector_store is mock_store
