from abc import ABC, abstractmethod

from ragforge.domain.models import GenerationRequest, GenerationResponse


class BaseLLMProvider(ABC):
    """Abstract port for asynchronous large language model generation providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Normalized string identifier of the provider (e.g. 'groq', 'openai')."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a completion for the specified generation request.

        Args:
            request: Provider-agnostic generation request containing prompt, system
                instructions, sampling hyperparameters, and execution configuration.

        Returns:
            Structured GenerationResponse model containing generated text, model
            identifier, provider provenance, and token usage metadata.

        Raises:
            LLMError: Base exception for any generation failure, mapped into
                appropriate domain exception subclasses.
        """
