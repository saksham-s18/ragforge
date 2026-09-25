"""Unit tests for Groq and OpenAI LLM provider adapters."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from groq import (
    APIConnectionError as GroqConnectionError,
)
from groq import (
    APIError as GroqAPIError,
)
from groq import (
    APITimeoutError as GroqTimeoutError,
)
from groq import (
    AuthenticationError as GroqAuthError,
)
from groq import (
    BadRequestError as GroqBadRequestError,
)
from groq import (
    InternalServerError as GroqInternalServerError,
)
from groq import (
    RateLimitError as GroqRateLimitError,
)
from openai import (
    APIConnectionError as OpenAIConnectionError,
)
from openai import (
    APIError as OpenAIAPIError,
)
from openai import (
    APITimeoutError as OpenAITimeoutError,
)
from openai import (
    AuthenticationError as OpenAIAuthError,
)
from openai import (
    BadRequestError as OpenAIBadRequestError,
)
from openai import (
    InternalServerError as OpenAIInternalServerError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)

from ragforge.adapters.llm.groq import GroqLLMProvider
from ragforge.adapters.llm.openai import OpenAILLMProvider
from ragforge.domain.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMConnectionError,
    LLMError,
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ragforge.domain.models import GenerationRequest

# ---------------------------------------------------------------------------
# Groq Provider Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_adapter_success() -> None:
    """Verify Groq adapter successfully generates response with mocked client."""
    provider = GroqLLMProvider(
        api_key="gsk_test_mock_key_12345",
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=500,
    )

    # Mock SDK response
    mock_choice = MagicMock()
    mock_choice.message.content = "Groq generated answer."
    mock_completion = MagicMock()
    mock_completion.model = "llama-3.3-70b-versatile"
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 45
    mock_completion.usage.completion_tokens = 20
    mock_completion.usage.total_tokens = 65

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    provider._client = mock_client

    request = GenerationRequest(
        user_prompt="What is RAG?",
        system_instruction="Be concise.",
    )

    response = await provider.generate(request)

    assert response.text == "Groq generated answer."
    assert response.provider == "groq"
    assert response.model == "llama-3.3-70b-versatile"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 45
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 65

    mock_client.chat.completions.create.assert_awaited_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["max_tokens"] == 500
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "What is RAG?"},
    ]


def test_groq_missing_api_key() -> None:
    """Verify Groq adapter raises LLMConfigurationError when API key is missing."""
    with pytest.raises(LLMConfigurationError) as exc_info:
        GroqLLMProvider(api_key="", model="llama-3.3-70b-versatile")

    assert "GROQ_API_KEY" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError):
        GroqLLMProvider(api_key=None, model="llama-3.3-70b-versatile")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk_error", "expected_domain_error"),
    [
        (
            GroqAuthError("Invalid API key", response=MagicMock(status_code=401), body={}),
            LLMAuthenticationError,
        ),
        (
            GroqRateLimitError(
                "Rate limit exceeded",
                response=MagicMock(status_code=429),
                body={},
            ),
            LLMRateLimitError,
        ),
        (
            GroqTimeoutError(request=MagicMock()),
            LLMTimeoutError,
        ),
        (
            GroqConnectionError(request=MagicMock()),
            LLMConnectionError,
        ),
        (
            GroqBadRequestError("Bad request", response=MagicMock(status_code=400), body={}),
            LLMInvalidRequestError,
        ),
        (
            GroqInternalServerError(
                "Server error",
                response=MagicMock(status_code=500),
                body={},
            ),
            LLMProviderUnavailableError,
        ),
        (
            GroqAPIError("Unclassified API error", request=MagicMock(), body={}),
            LLMError,
        ),
    ],
)
async def test_groq_error_mapping(
    sdk_error: Exception, expected_domain_error: type[Exception]
) -> None:
    """Verify Groq SDK exceptions are accurately translated into domain exceptions."""
    provider = GroqLLMProvider(api_key="gsk_test_key", model="llama-3.3-70b-versatile")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=sdk_error)
    provider._client = mock_client

    request = GenerationRequest(user_prompt="Hello")

    with pytest.raises(expected_domain_error):
        await provider.generate(request)


# ---------------------------------------------------------------------------
# OpenAI Provider Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_adapter_success() -> None:
    """Verify OpenAI adapter successfully generates response with mocked client."""
    provider = OpenAILLMProvider(
        api_key="sk-test-mock-key-12345",
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=300,
    )

    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI generated answer."
    mock_completion = MagicMock()
    mock_completion.model = "gpt-4o-mini"
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 30
    mock_completion.usage.completion_tokens = 15
    mock_completion.usage.total_tokens = 45

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    provider._client = mock_client

    request = GenerationRequest(
        user_prompt="Explain indexing.",
        system_instruction="Be technical.",
        temperature=0.5,
        max_tokens=250,
    )

    response = await provider.generate(request)

    assert response.text == "OpenAI generated answer."
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 30

    mock_client.chat.completions.create.assert_awaited_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o-mini"
    # Overridden in request:
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_tokens"] == 250


def test_openai_missing_api_key() -> None:
    """Verify OpenAI adapter raises LLMConfigurationError when API key is missing."""
    with pytest.raises(LLMConfigurationError) as exc_info:
        OpenAILLMProvider(api_key="", model="gpt-4o-mini")

    assert "OPENAI_API_KEY" in str(exc_info.value)

    with pytest.raises(LLMConfigurationError):
        OpenAILLMProvider(api_key=None, model="gpt-4o-mini")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk_error", "expected_domain_error"),
    [
        (
            OpenAIAuthError("Invalid credentials", response=MagicMock(status_code=401), body={}),
            LLMAuthenticationError,
        ),
        (
            OpenAIRateLimitError(
                "Rate limit hit",
                response=MagicMock(status_code=429),
                body={},
            ),
            LLMRateLimitError,
        ),
        (
            OpenAITimeoutError(request=MagicMock()),
            LLMTimeoutError,
        ),
        (
            OpenAIConnectionError(request=MagicMock()),
            LLMConnectionError,
        ),
        (
            OpenAIBadRequestError("Malformed body", response=MagicMock(status_code=400), body={}),
            LLMInvalidRequestError,
        ),
        (
            OpenAIInternalServerError(
                "Internal server error",
                response=MagicMock(status_code=500),
                body={},
            ),
            LLMProviderUnavailableError,
        ),
        (
            OpenAIAPIError("Generic OpenAI API error", request=MagicMock(), body={}),
            LLMError,
        ),
    ],
)
async def test_openai_error_mapping(
    sdk_error: Exception, expected_domain_error: type[Exception]
) -> None:
    """Verify OpenAI SDK exceptions are accurately translated into domain exceptions."""
    provider = OpenAILLMProvider(api_key="sk-test-key", model="gpt-4o-mini")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=sdk_error)
    provider._client = mock_client

    request = GenerationRequest(user_prompt="Hello")

    with pytest.raises(expected_domain_error):
        await provider.generate(request)


# ---------------------------------------------------------------------------
# Credential Safety & Log Sanitization (Requirement S)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_credentials_never_appear_in_errors_or_repr() -> None:
    """Verify secret API keys are never exposed in string representations or error details."""
    secret_key = "gsk_super_secret_production_key_xyz987"
    provider = GroqLLMProvider(api_key=secret_key, model="llama-3.3-70b-versatile")

    # Check string representation of the adapter
    assert secret_key not in repr(provider)
    assert secret_key not in str(provider)

    # Trigger an error and verify the secret key is not in the raised exception string
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=GroqAuthError(
            f"Unauthorized: key {secret_key} is invalid",
            response=MagicMock(status_code=401),
            body={},
        )
    )
    provider._client = mock_client

    with pytest.raises(LLMAuthenticationError) as exc_info:
        await provider.generate(GenerationRequest(user_prompt="Test"))

    # The exception message should be sanitized or generic, not exposing the secret
    assert secret_key not in str(exc_info.value)
