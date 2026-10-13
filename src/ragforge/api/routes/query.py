import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ragforge.api.dependencies import get_rag_service
from ragforge.core.config import Settings, get_settings
from ragforge.domain.enums import SearchStrategy
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
from ragforge.services.rag import RAGGenerationService

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
    search_strategy: SearchStrategy | None = Field(
        default=None,
        description="Optional retrieval search strategy ('dense', 'sparse', 'hybrid'). "
        "Defaults to dense.",
    )


class QueryResponse(BaseModel):
    """Response payload containing generated answer and provenance citations."""

    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    provider: str
    model: str


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate grounded RAG answer with source citations",
)
async def query_documents(
    payload: QueryRequest,
    rag_service: Annotated[RAGGenerationService, Depends(get_rag_service)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryResponse:
    """Execute retrieval-augmented answer synthesis for a user question."""
    if not payload.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    if (
        payload.search_strategy in (SearchStrategy.HYBRID, SearchStrategy.SPARSE)
        and not settings.hybrid_enabled
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{payload.search_strategy.value}' retrieval is disabled in "
                "application configuration."
            ),
        )

    try:
        rag_response = await rag_service.generate_answer(
            question=payload.question,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            search_strategy=payload.search_strategy,
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


__all__ = ["QueryRequest", "QueryResponse", "get_rag_service", "query_documents", "router"]
