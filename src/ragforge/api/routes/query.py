import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ragforge.adapters.embeddings import (
    DeterministicEmbeddingProvider,
    FastEmbedProvider,
)
from ragforge.adapters.llm import GroqLLMProvider, OpenAILLMProvider
from ragforge.adapters.vector_stores import QdrantVectorStore
from ragforge.core.config import Settings, get_settings
from ragforge.domain.exceptions import (
    AllLLMProvidersFailedError,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMTransientError,
)
from ragforge.domain.models import SourceReference
from ragforge.ports.embeddings import BaseEmbeddingProvider
from ragforge.ports.llm import BaseLLMProvider
from ragforge.services.llm_router import LLMRouter
from ragforge.services.rag import RAGGenerationService
from ragforge.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])


class QueryRequest(BaseModel):
    """Input payload for RAG question-answering endpoint."""

    question: str = Field(
        ...,
        description="User question to answer against indexed document context",
        min_length=1,
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of nearest chunks to retrieve for context",
    )
    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity cutoff for context chunks",
    )


class QueryResponse(BaseModel):
    """Response payload containing generated answer and provenance citations."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    provider: str
    model: str


def get_rag_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGGenerationService:
    """Dependency provider instantiating the default RAG generation service."""
    # 1. Embedding Provider
    embedding_provider: BaseEmbeddingProvider
    if settings.embedding_provider == "fastembed":
        embedding_provider = FastEmbedProvider(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        )
    else:
        embedding_provider = DeterministicEmbeddingProvider(
            dimension=settings.qdrant_vector_dimension
        )

    # 2. Vector Store & Retrieval Service
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        dimension=settings.qdrant_vector_dimension,
    )
    retrieval_service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )

    # 3. Primary LLM Provider
    primary_provider: BaseLLMProvider
    if settings.llm_provider == "openai":
        primary_provider = OpenAILLMProvider(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )
    else:
        primary_provider = GroqLLMProvider(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )

    # 4. Optional Fallback LLM Provider
    fallback_provider: BaseLLMProvider | None = None
    if settings.llm_fallback_provider == "openai" and settings.openai_api_key:
        fallback_provider = OpenAILLMProvider(
            api_key=settings.openai_api_key,
            model=settings.llm_fallback_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )
    elif settings.llm_fallback_provider == "groq" and settings.groq_api_key:
        fallback_provider = GroqLLMProvider(
            api_key=settings.groq_api_key,
            model=settings.llm_fallback_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )

    # 5. Router and RAG Service
    llm_router = LLMRouter(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )
    return RAGGenerationService(
        retrieval_service=retrieval_service,
        llm_router=llm_router,
        default_top_k=5,
        default_temperature=settings.llm_temperature,
        default_max_tokens=settings.llm_max_tokens,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate grounded RAG answer with source citations",
)
async def query_documents(
    payload: QueryRequest,
    rag_service: Annotated[RAGGenerationService, Depends(get_rag_service)],
) -> QueryResponse:
    """Execute retrieval-augmented answer synthesis for a user question."""
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    try:
        rag_response = await rag_service.generate_answer(
            question=payload.question,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )
        return QueryResponse(
            answer=rag_response.answer,
            sources=rag_response.sources,
            provider=rag_response.provider,
            model=rag_response.model,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except LLMConfigurationError as exc:
        logger.error("LLM configuration error during query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM provider configuration error.",
        ) from exc
    except LLMAuthenticationError as exc:
        logger.error("LLM authentication failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed with configured LLM provider.",
        ) from exc
    except AllLLMProvidersFailedError as exc:
        logger.error("All LLM providers failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="All configured LLM providers are currently unavailable.",
        ) from exc
    except (LLMRateLimitError, LLMTransientError) as exc:
        logger.error("Transient error during LLM generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is currently unavailable or rate-limited.",
        ) from exc
    except LLMInvalidRequestError as exc:
        logger.error("Invalid request rejected by LLM: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Generation request was rejected by the LLM provider.",
        ) from exc
    except LLMError as exc:
        logger.error("Unclassified LLM error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate answer from upstream LLM provider.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during RAG query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during RAG processing.",
        ) from exc
