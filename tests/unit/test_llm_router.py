"""Unit tests for the LLM Provider Router and Fallback Mechanism."""

from unittest.mock import AsyncMock

import pytest

from ragforge.domain.exceptions import (
    AllLLMProvidersFailedError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ragforge.domain.models import GenerationRequest, GenerationResponse
from ragforge.ports.llm import BaseLLMProvider
from ragforge.services.llm_router import LLMRouter


class DummyProvider(BaseLLMProvider):
    """Test stub implementing BaseLLMProvider."""

    def __init__(self, name: str, model: str = "dummy-model") -> None:
        self._name = name
        self._model = model
        self.generate_mock = AsyncMock()

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def default_model(self) -> str:
        return self._model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        return await self.generate_mock(request)  # type: ignore[no-any-return]


@pytest.fixture
def groq_provider() -> DummyProvider:
    return DummyProvider(name="groq", model="llama-3.3-70b-versatile")


@pytest.fixture
def openai_provider() -> DummyProvider:
    return DummyProvider(name="openai", model="gpt-4o-mini")


# ---------------------------------------------------------------------------
# Router Provider Selection & Routing Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_selects_groq_as_primary(
    groq_provider: DummyProvider,
    openai_provider: DummyProvider,
) -> None:
    """Verify router dispatches to Groq when configured as primary."""
    groq_provider.generate_mock.return_value = GenerationResponse(
        text="Groq response",
        provider="groq",
        model="llama-3.3-70b-versatile",
    )

    router = LLMRouter(primary_provider=groq_provider, fallback_provider=openai_provider)
    request = GenerationRequest(user_prompt="Hello")

    response = await router.generate(request)

    assert response.provider == "groq"
    assert response.text == "Groq response"
    groq_provider.generate_mock.assert_awaited_once_with(request)
    openai_provider.generate_mock.assert_not_called()


@pytest.mark.asyncio
async def test_router_selects_openai_as_primary(
    groq_provider: DummyProvider,
    openai_provider: DummyProvider,
) -> None:
    """Verify router dispatches to OpenAI when configured as primary."""
    openai_provider.generate_mock.return_value = GenerationResponse(
        text="OpenAI primary response",
        provider="openai",
        model="gpt-4o-mini",
    )

    router = LLMRouter(primary_provider=openai_provider, fallback_provider=groq_provider)
    request = GenerationRequest(user_prompt="Hello")

    response = await router.generate(request)

    assert response.provider == "openai"
    assert response.text == "OpenAI primary response"
    openai_provider.generate_mock.assert_awaited_once_with(request)
    groq_provider.generate_mock.assert_not_called()


@pytest.mark.asyncio
async def test_groq_success_means_openai_not_called(
    groq_provider: DummyProvider,
    openai_provider: DummyProvider,
) -> None:
    """Verify that when primary provider succeeds, fallback provider is never called."""
    groq_provider.generate_mock.return_value = GenerationResponse(
        text="Success from Groq",
        provider="groq",
        model="llama-3.3-70b-versatile",
    )

    router = LLMRouter(primary_provider=groq_provider, fallback_provider=openai_provider)
    request = GenerationRequest(user_prompt="Test prompt")

    await router.generate(request)

    groq_provider.generate_mock.assert_awaited_once()
    openai_provider.generate_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback Behavior Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transient_error",
    [
        LLMRateLimitError("Rate limit reached on primary", provider="groq"),
        LLMTimeoutError("Timed out contacting primary", provider="groq"),
        LLMConnectionError("Network drop", provider="groq"),
        LLMProviderUnavailableError("Capacity issue 503", provider="groq"),
    ],
)
async def test_eligible_failure_triggers_fallback(
    groq_provider: DummyProvider,
    openai_provider: DummyProvider,
    transient_error: Exception,
) -> None:
    """Verify transient / capacity / rate limit errors trigger fallback provider."""
    groq_provider.generate_mock.side_effect = transient_error
    openai_provider.generate_mock.return_value = GenerationResponse(
        text="Fallback answer from OpenAI",
        provider="openai",
        model="gpt-4o-mini",
    )

    router = LLMRouter(primary_provider=groq_provider, fallback_provider=openai_provider)
    request = GenerationRequest(user_prompt="Explain RAG")

    response = await router.generate(request)

    assert response.provider == "openai"
    assert response.text == "Fallback answer from OpenAI"
    groq_provider.generate_mock.assert_awaited_once()
    openai_provider.generate_mock.assert_awaited_once_with(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "non_eligible_error",
    [
        LLMAuthenticationError("Bad API key", provider="groq"),
        LLMConfigurationError("Missing config", provider="groq"),
        LLMInvalidRequestError("Context window exceeded", provider="groq"),
        ValueError("Local argument validation failed"),
    ],
)
async def test_non_eligible_failure_does_not_trigger_fallback(
    groq_provider: DummyProvider,
    openai_provider: DummyProvider,
    non_eligible_error: Exception,
) -> None:
    """Verify non-transient, configuration, or programming errors fail
    immediately without invoking fallback."""
    groq_provider.generate_mock.side_effect = non_eligible_error

    router = LLMRouter(primary_provider=groq_provider, fallback_provider=openai_provider)
    request = GenerationRequest(user_prompt="Test")

    with pytest.raises(type(non_eligible_error)):
        await router.generate(request)

    groq_provider.generate_mock.assert_awaited_once()
    openai_provider.generate_mock.assert_not_called()


@pytest.mark.asyncio
async def test_both_providers_failing_raises_all_providers_failed(
    groq_provider: DummyProvider,
    openai_provider: DummyProvider,
) -> None:
    """Verify that if both primary and fallback fail, AllLLMProvidersFailedError is raised."""
    groq_provider.generate_mock.side_effect = LLMTimeoutError("Groq timeout", provider="groq")
    openai_provider.generate_mock.side_effect = LLMRateLimitError(
        "OpenAI rate limited", provider="openai"
    )

    router = LLMRouter(primary_provider=groq_provider, fallback_provider=openai_provider)
    request = GenerationRequest(user_prompt="Question")

    with pytest.raises(AllLLMProvidersFailedError) as exc_info:
        await router.generate(request)

    err = exc_info.value
    assert "groq" in str(err)
    assert "openai" in str(err)
    assert err.primary_error is not None
    assert err.fallback_error is not None


@pytest.mark.asyncio
async def test_no_fallback_configured_raises_primary_error_directly(
    groq_provider: DummyProvider,
) -> None:
    """Verify that when no fallback is configured, transient primary errors are raised directly."""
    groq_provider.generate_mock.side_effect = LLMRateLimitError("Rate limit", provider="groq")

    router = LLMRouter(primary_provider=groq_provider, fallback_provider=None)
    request = GenerationRequest(user_prompt="Question")

    with pytest.raises(LLMRateLimitError):
        await router.generate(request)
