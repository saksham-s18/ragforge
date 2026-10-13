"""Unit tests for the FastAPI RAG query endpoint (POST /api/v1/query)."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from ragforge.api.app import create_app
from ragforge.api.routes.query import get_rag_service
from ragforge.core.config import Settings
from ragforge.domain.enums import SearchStrategy
from ragforge.domain.exceptions import (
    AllLLMProvidersFailedError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
)
from ragforge.domain.models import RAGResponse, SourceReference
from ragforge.services.rag import RAGGenerationService


@pytest.fixture
def mock_rag_service() -> AsyncMock:
    """Fixture providing a mock RAGGenerationService."""
    mock = AsyncMock(spec=RAGGenerationService)
    mock.generate_answer.return_value = RAGResponse(
        question="What is RAGForge?",
        answer="RAGForge is a production-oriented RAG framework.",
        sources=[
            SourceReference(
                document_id="doc-1",
                chunk_id="chunk-101",
                source_path="README.md",
                section_header="Overview",
                start_char=0,
                end_char=100,
                score=0.91,
                content_snippet="RAGForge is an agentic framework...",
            )
        ],
        provider="groq",
        model="llama-3.3-70b-versatile",
    )
    return mock


@pytest.fixture
def test_app(mock_rag_service: AsyncMock) -> tuple:
    """Application configured with dependency overrides for offline query testing."""
    settings = Settings(env="test", debug=True, hybrid_enabled=False)
    app = create_app(settings=settings)
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    return app, mock_rag_service


@pytest.fixture
def hybrid_test_app(mock_rag_service: AsyncMock) -> tuple:
    """Application configured with hybrid retrieval enabled for query testing."""
    settings = Settings(env="test", debug=True, hybrid_enabled=True)
    app = create_app(settings=settings)
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    return app, mock_rag_service


# ---------------------------------------------------------------------------
# Endpoint Success Tests (Requirement Q)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_endpoint_success(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns HTTP 200 and structured response."""
    app, mock_rag = test_app

    mock_rag.generate_answer.return_value = RAGResponse(
        question="What is RAGForge?",
        answer="RAGForge is a production-oriented RAG framework.",
        sources=[
            SourceReference(
                document_id="doc-1",
                chunk_id="chunk-101",
                source_path="README.md",
                section_header="Overview",
                start_char=0,
                end_char=100,
                score=0.91,
                content_snippet="RAGForge is an agentic framework...",
            )
        ],
        provider="groq",
        model="llama-3.3-70b-versatile",
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is RAGForge?", "top_k": 3},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "RAGForge is a production-oriented RAG framework."
    assert data["provider"] == "groq"
    assert data["model"] == "llama-3.3-70b-versatile"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "doc-1"
    assert data["sources"][0]["chunk_id"] == "chunk-101"
    assert data["sources"][0]["source_path"] == "README.md"
    assert data["sources"][0]["score"] == 0.91


# ---------------------------------------------------------------------------
# Endpoint Failure & Validation Tests (Requirement R)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_endpoint_empty_question(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns 400 for empty or whitespace-only questions."""
    app, _ = test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Pydantic min_length catches empty string -> 422
        resp_empty = await client.post("/api/v1/query", json={"question": ""})
        assert resp_empty.status_code == 422

        # Whitespace-only string caught by route check -> 400
        resp_spaces = await client.post("/api/v1/query", json={"question": "    "})
        assert resp_spaces.status_code == 400


@pytest.mark.asyncio
async def test_query_endpoint_missing_payload(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns 422 when required question is missing."""
    app, _ = test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_endpoint_config_error(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns 500 on LLMConfigurationError."""
    app, mock_rag = test_app
    mock_rag.generate_answer.side_effect = LLMConfigurationError("Missing API key", provider="groq")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"question": "Test question"})

    assert response.status_code == 500
    assert "configuration" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_endpoint_auth_error(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns 500 on LLMAuthenticationError."""
    app, mock_rag = test_app
    mock_rag.generate_answer.side_effect = LLMAuthenticationError(
        "Bad credentials", provider="groq"
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"question": "Test question"})

    assert response.status_code == 500
    assert "authentication" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_endpoint_all_providers_failed(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns 503 when all providers fail."""
    app, mock_rag = test_app
    mock_rag.generate_answer.side_effect = AllLLMProvidersFailedError(
        primary_provider="groq",
        fallback_provider="openai",
        primary_error=Exception("primary down"),
        fallback_error=Exception("fallback down"),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"question": "Test question"})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_endpoint_rate_limit(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns 503 on unhandled rate limit / transient error."""
    app, mock_rag = test_app
    mock_rag.generate_answer.side_effect = LLMRateLimitError("Too many requests", provider="groq")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"question": "Test question"})

    assert response.status_code == 503
    assert "rate-limited" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_endpoint_invalid_request(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns 422 when provider rejects request."""
    app, mock_rag = test_app
    mock_rag.generate_answer.side_effect = LLMInvalidRequestError(
        "Invalid model parameters", provider="groq"
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"question": "Test question"})

    assert response.status_code == 422
    assert "rejected" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_query_endpoint_upstream_llm_error(test_app: tuple) -> None:
    """Verify POST /api/v1/query returns 502 for unclassified upstream LLM error."""
    app, mock_rag = test_app
    mock_rag.generate_answer.side_effect = LLMError("Upstream failure", provider="groq")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/query", json={"question": "Test question"})

    assert response.status_code == 502
    assert "upstream" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Search Strategy & Hybrid Retrieval Tests (Stage 11 Step 4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_endpoint_omitted_search_strategy_routes_default(test_app: tuple) -> None:
    """Verify omitting search_strategy preserves backward compatibility and passes None."""
    app, mock_rag = test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is RAGForge?", "top_k": 3},
        )

    assert response.status_code == 200
    mock_rag.generate_answer.assert_awaited_once_with(
        question="What is RAGForge?",
        top_k=3,
        score_threshold=None,
        search_strategy=None,
    )


@pytest.mark.asyncio
async def test_query_endpoint_explicit_dense_strategy(test_app: tuple) -> None:
    """Verify explicit 'dense' search_strategy routes to DENSE even when hybrid is disabled."""
    app, mock_rag = test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={
                "question": "What is RAGForge?",
                "top_k": 5,
                "search_strategy": "dense",
            },
        )

    assert response.status_code == 200
    mock_rag.generate_answer.assert_awaited_once_with(
        question="What is RAGForge?",
        top_k=5,
        score_threshold=None,
        search_strategy=SearchStrategy.DENSE,
    )


@pytest.mark.asyncio
async def test_query_endpoint_hybrid_disabled_rejects_hybrid(test_app: tuple) -> None:
    """Verify requesting 'hybrid' when hybrid is disabled returns 400 Bad Request."""
    app, mock_rag = test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is RAGForge?", "search_strategy": "hybrid"},
        )

    assert response.status_code == 400
    assert "hybrid' retrieval is disabled" in response.json()["detail"]
    mock_rag.generate_answer.assert_not_called()


@pytest.mark.asyncio
async def test_query_endpoint_hybrid_disabled_rejects_sparse(test_app: tuple) -> None:
    """Verify requesting 'sparse' when hybrid is disabled returns 400 Bad Request."""
    app, mock_rag = test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is RAGForge?", "search_strategy": "sparse"},
        )

    assert response.status_code == 400
    assert "sparse' retrieval is disabled" in response.json()["detail"]
    mock_rag.generate_answer.assert_not_called()


@pytest.mark.asyncio
async def test_query_endpoint_hybrid_enabled_routes_hybrid(hybrid_test_app: tuple) -> None:
    """Verify requesting 'hybrid' when hybrid is enabled routes to SearchStrategy.HYBRID."""
    app, mock_rag = hybrid_test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is RAGForge?", "search_strategy": "hybrid"},
        )

    assert response.status_code == 200
    mock_rag.generate_answer.assert_awaited_once_with(
        question="What is RAGForge?",
        top_k=5,
        score_threshold=None,
        search_strategy=SearchStrategy.HYBRID,
    )


@pytest.mark.asyncio
async def test_query_endpoint_hybrid_enabled_routes_sparse(hybrid_test_app: tuple) -> None:
    """Verify requesting 'sparse' when hybrid is enabled routes to SearchStrategy.SPARSE."""
    app, mock_rag = hybrid_test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is RAGForge?", "search_strategy": "sparse"},
        )

    assert response.status_code == 200
    mock_rag.generate_answer.assert_awaited_once_with(
        question="What is RAGForge?",
        top_k=5,
        score_threshold=None,
        search_strategy=SearchStrategy.SPARSE,
    )


@pytest.mark.asyncio
async def test_query_endpoint_hybrid_enabled_omitted_strategy_remains_dense(
    hybrid_test_app: tuple,
) -> None:
    """Verify omitting search_strategy when hybrid is enabled continues to pass None."""
    app, mock_rag = hybrid_test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is RAGForge?"},
        )

    assert response.status_code == 200
    mock_rag.generate_answer.assert_awaited_once_with(
        question="What is RAGForge?",
        top_k=5,
        score_threshold=None,
        search_strategy=None,
    )


@pytest.mark.asyncio
async def test_query_endpoint_invalid_search_strategy(test_app: tuple) -> None:
    """Verify invalid search_strategy string is rejected with 422 Unprocessable Content."""
    app, mock_rag = test_app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/query",
            json={"question": "What is RAGForge?", "search_strategy": "neural_super_search"},
        )

    assert response.status_code == 422
    mock_rag.generate_answer.assert_not_called()
