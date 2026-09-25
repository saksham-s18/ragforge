import logging

from ragforge.domain.exceptions import (
    AllLLMProvidersFailedError,
    LLMError,
    LLMTransientError,
)
from ragforge.domain.models import GenerationRequest, GenerationResponse
from ragforge.ports.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMRouter:
    """Intelligent routing and controlled fallback orchestrator for LLM generation providers.

    Dispatches generation requests to the primary LLM provider (e.g. Groq) and seamlessly
    fails over to a configured fallback provider (e.g. OpenAI) exclusively when transient,
    availability, rate-limit, or connection errors occur.
    """

    def __init__(
        self,
        primary_provider: BaseLLMProvider,
        fallback_provider: BaseLLMProvider | None = None,
    ) -> None:
        """Initialize the router with primary and optional fallback providers.

        Args:
            primary_provider: Main LLM provider adapter (e.g. GroqLLMProvider).
            fallback_provider: Optional backup LLM provider adapter (e.g. OpenAILLMProvider).
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider

    def is_fallback_eligible(self, exc: Exception) -> bool:
        """Determine whether an exception qualifies for automated fallback routing.

        Fallback-eligible errors include transient network failures, timeouts, rate limits,
        and upstream service outages (5xx). Programming, authentication, and bad request
        errors are NOT fallback eligible and must fail fast.
        """
        if isinstance(exc, LLMTransientError):
            return True
        if isinstance(exc, LLMError):
            return getattr(exc, "is_fallback_eligible", False)
        return False

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Execute text generation with primary provider and bounded fallback failover.

        Args:
            request: Provider-agnostic generation parameters.

        Returns:
            Structured GenerationResponse model containing generated text and provider provenance.

        Raises:
            LLMError: When primary provider fails without eligible fallback or with no fallback.
            AllLLMProvidersFailedError: When both primary and fallback providers fail.
        """
        primary_name = self.primary_provider.provider_name
        logger.debug(
            "Routing generation request to primary provider '%s'",
            primary_name,
        )

        try:
            response = await self.primary_provider.generate(request)
            logger.debug(
                "Primary provider '%s' succeeded (model: %s)",
                primary_name,
                response.model,
            )
            return response
        except Exception as primary_exc:
            if not self.is_fallback_eligible(primary_exc) or self.fallback_provider is None:
                # Fast fail for non-transient errors, programming bugs, or when no fallback exists
                logger.debug(
                    "Primary provider '%s' raised non-fallback eligible error or no fallback "
                    "configured: %s",
                    primary_name,
                    primary_exc,
                )
                raise

            fallback_name = self.fallback_provider.provider_name
            error_desc = getattr(primary_exc, "message", str(primary_exc))
            logger.warning(
                "Primary LLM provider '%s' failed with transient error: %s. "
                "Triggering fallback routing to '%s'.",
                primary_name,
                error_desc,
                fallback_name,
            )

            try:
                fallback_response = await self.fallback_provider.generate(request)
                logger.info(
                    "Fallback provider '%s' succeeded following primary failure (model: %s)",
                    fallback_name,
                    fallback_response.model,
                )
                return fallback_response
            except Exception as fallback_exc:
                logger.error(
                    "Fallback provider '%s' also failed: %s",
                    fallback_name,
                    fallback_exc,
                )
                raise AllLLMProvidersFailedError(
                    f"Both primary provider '{primary_name}' and fallback provider "
                    f"'{fallback_name}' failed to generate an answer.",
                    primary_error=primary_exc,
                    fallback_error=fallback_exc,
                ) from fallback_exc
