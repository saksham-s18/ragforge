import logging
import re
from typing import Any

import groq
from groq import AsyncGroq

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

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class GroqLLMProvider(BaseLLMProvider):
    """Production asynchronous LLM adapter integrating Groq Cloud inference."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_GROQ_MODEL,
        temperature: float = 0.0,
        max_tokens: int | None = 1024,
        timeout: float = 30.0,
        client: AsyncGroq | None = None,
    ) -> None:
        """Initialize the Groq LLM provider.

        Args:
            api_key: Groq Cloud API key. If omitted, checks client or raises error.
            model: Default model identifier (defaults to llama-3.3-70b-versatile).
            temperature: Default sampling temperature (0.0 to 2.0).
            max_tokens: Default maximum output tokens to generate.
            timeout: HTTP request timeout in seconds.
            client: Optional injected AsyncGroq client instance (for testing/mocking).

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
            self._client = AsyncGroq(api_key=api_key.strip(), timeout=timeout)
        else:
            raise LLMConfigurationError(
                "Groq API key (GROQ_API_KEY) must be provided via settings or constructor.",
                provider="groq",
            )

    @property
    def provider_name(self) -> str:
        """Normalized string identifier for the provider."""
        return "groq"

    def _sanitize(self, message: str) -> str:
        """Redact any sensitive credentials from error messages."""
        sanitized = message
        if self._api_key and len(self._api_key) > 6:
            sanitized = sanitized.replace(self._api_key, "[REDACTED_API_KEY]")
        sanitized = re.sub(r"gsk_[a-zA-Z0-9]{20,}", "[REDACTED_API_KEY]", sanitized)
        return sanitized

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion for the given request using Groq Cloud.

        Args:
            request: Provider-agnostic generation request.

        Returns:
            Structured GenerationResponse model.

        Raises:
            LLMAuthenticationError: When API key is invalid or rejected.
            LLMRateLimitError: When Groq rate limits are exceeded.
            LLMTimeoutError: When generation request times out.
            LLMConnectionError: When network connectivity fails.
            LLMProviderUnavailableError: When Groq internal server error occurs (5xx).
            LLMInvalidRequestError: When request parameters or prompt are rejected.
            LLMError: For any other unclassified Groq API errors.
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
            "Dispatching generation request to Groq (model=%s, temp=%.2f, max_tokens=%s)",
            effective_model,
            effective_temp,
            effective_max_tokens,
        )

        try:
            completion = await self._client.chat.completions.create(**create_kwargs)
        except groq.AuthenticationError as exc:
            msg = self._sanitize(f"Groq authentication failed: {exc.message}")
            raise LLMAuthenticationError(msg, provider="groq") from exc
        except groq.RateLimitError as exc:
            msg = self._sanitize(f"Groq rate limit exceeded: {exc.message}")
            raise LLMRateLimitError(msg, provider="groq") from exc
        except groq.APITimeoutError as exc:
            msg = self._sanitize(f"Groq request timed out: {exc}")
            raise LLMTimeoutError(msg, provider="groq") from exc
        except groq.APIConnectionError as exc:
            msg = self._sanitize(f"Failed to connect to Groq endpoint: {exc.message}")
            raise LLMConnectionError(msg, provider="groq") from exc
        except groq.InternalServerError as exc:
            msg = self._sanitize(f"Groq service error (500): {exc.message}")
            raise LLMProviderUnavailableError(msg, provider="groq") from exc
        except groq.BadRequestError as exc:
            msg = self._sanitize(f"Groq bad request: {exc.message}")
            raise LLMInvalidRequestError(msg, provider="groq") from exc
        except groq.APIStatusError as exc:
            status_code = getattr(exc, "status_code", 500)
            msg = self._sanitize(f"Groq status error ({status_code}): {exc.message}")
            if status_code >= 500:
                raise LLMProviderUnavailableError(msg, provider="groq") from exc
            raise LLMError(msg, provider="groq") from exc
        except groq.GroqError as exc:
            msg = self._sanitize(f"Groq client error: {exc}")
            raise LLMError(msg, provider="groq") from exc
        except Exception as exc:
            msg = self._sanitize(f"Unexpected error executing Groq generation: {exc}")
            raise LLMError(msg, provider="groq") from exc

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
            provider="groq",
            model=model_name,
            usage=usage,
        )
