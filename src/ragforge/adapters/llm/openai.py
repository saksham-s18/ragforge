import logging
import re
from typing import Any

import openai
from openai import AsyncOpenAI

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
from ragforge.domain.models import GenerationRequest, GenerationResponse, LLMUsage
from ragforge.ports.llm import BaseLLMProvider

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAILLMProvider(BaseLLMProvider):
    """Production asynchronous LLM adapter integrating OpenAI inference."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = 0.0,
        max_tokens: int | None = 1024,
        timeout: float = 30.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Initialize the OpenAI LLM provider.

        Args:
            api_key: OpenAI API key. If omitted, checks client or raises error.
            model: Default model identifier (defaults to gpt-4o-mini).
            temperature: Default sampling temperature (0.0 to 2.0).
            max_tokens: Default maximum output tokens to generate.
            timeout: HTTP request timeout in seconds.
            client: Optional injected AsyncOpenAI client instance (for testing/mocking).

        Raises:
            LLMConfigurationError: If neither an API key nor a client instance is provided.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = api_key

        if client is not None:
            self._client = client
        elif api_key and api_key.strip():
            self._client = AsyncOpenAI(api_key=api_key.strip(), timeout=timeout)
        else:
            raise LLMConfigurationError(
                "OpenAI API key (OPENAI_API_KEY) must be provided via settings or constructor.",
                provider="openai",
            )

    @property
    def provider_name(self) -> str:
        """Normalized string identifier for the provider."""
        return "openai"

    def _sanitize(self, message: str) -> str:
        """Redact any sensitive credentials from error messages."""
        sanitized = message
        if self._api_key and len(self._api_key) > 6:
            sanitized = sanitized.replace(self._api_key, "[REDACTED_API_KEY]")
        sanitized = re.sub(r"sk-[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
        return sanitized

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion for the given request using OpenAI.

        Args:
            request: Provider-agnostic generation request.

        Returns:
            Structured GenerationResponse model.

        Raises:
            LLMAuthenticationError: When API key is invalid or rejected.
            LLMRateLimitError: When OpenAI rate limits are exceeded.
            LLMTimeoutError: When generation request times out.
            LLMConnectionError: When network connectivity fails.
            LLMProviderUnavailableError: When OpenAI internal server error occurs (5xx).
            LLMInvalidRequestError: When request parameters or prompt are rejected.
            LLMError: For any other unclassified OpenAI API errors.
        """
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        effective_model = request.model or self.model
        effective_temp = (
            request.temperature if request.temperature is not None else self.temperature
        )
        effective_max_tokens = (
            request.max_tokens if request.max_tokens is not None else self.max_tokens
        )

        create_kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": effective_temp,
        }
        if effective_max_tokens is not None:
            create_kwargs["max_tokens"] = effective_max_tokens

        logger.debug(
            "Dispatching generation request to OpenAI (model=%s, temp=%.2f, max_tokens=%s)",
            effective_model,
            effective_temp,
            effective_max_tokens,
        )

        try:
            completion = await self._client.chat.completions.create(**create_kwargs)
        except openai.AuthenticationError as exc:
            msg = self._sanitize(f"OpenAI authentication failed: {exc.message}")
            raise LLMAuthenticationError(msg, provider="openai") from exc
        except openai.RateLimitError as exc:
            msg = self._sanitize(f"OpenAI rate limit exceeded: {exc.message}")
            raise LLMRateLimitError(msg, provider="openai") from exc
        except openai.APITimeoutError as exc:
            msg = self._sanitize(f"OpenAI request timed out: {exc}")
            raise LLMTimeoutError(msg, provider="openai") from exc
        except openai.APIConnectionError as exc:
            msg = self._sanitize(f"Failed to connect to OpenAI endpoint: {exc.message}")
            raise LLMConnectionError(msg, provider="openai") from exc
        except openai.InternalServerError as exc:
            msg = self._sanitize(f"OpenAI service error (500): {exc.message}")
            raise LLMProviderUnavailableError(msg, provider="openai") from exc
        except openai.BadRequestError as exc:
            msg = self._sanitize(f"OpenAI bad request: {exc.message}")
            raise LLMInvalidRequestError(msg, provider="openai") from exc
        except openai.APIStatusError as exc:
            status_code = getattr(exc, "status_code", 500)
            msg = self._sanitize(f"OpenAI status error ({status_code}): {exc.message}")
            if status_code >= 500:
                raise LLMProviderUnavailableError(msg, provider="openai") from exc
            raise LLMError(msg, provider="openai") from exc
        except openai.OpenAIError as exc:
            msg = self._sanitize(f"OpenAI client error: {exc}")
            raise LLMError(msg, provider="openai") from exc
        except Exception as exc:
            msg = self._sanitize(f"Unexpected error executing OpenAI generation: {exc}")
            raise LLMError(msg, provider="openai") from exc

        content = ""
        if completion.choices and completion.choices[0].message:
            content = completion.choices[0].message.content or ""

        usage: LLMUsage | None = None
        if completion.usage:
            usage = LLMUsage(
                prompt_tokens=completion.usage.prompt_tokens or 0,
                completion_tokens=completion.usage.completion_tokens or 0,
                total_tokens=completion.usage.total_tokens or 0,
            )

        model_name = (
            completion.model
            if isinstance(getattr(completion, "model", None), str)
            else effective_model
        )

        return GenerationResponse(
            text=content,
            provider="openai",
            model=model_name,
            usage=usage,
        )
