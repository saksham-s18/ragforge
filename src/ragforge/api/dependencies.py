"""Application dependency providers and wiring for FastAPI and domain services."""

import logging
from typing import Annotated

from fastapi import Depends

from ragforge.adapters.embeddings import (
    DeterministicEmbeddingProvider,
    FastEmbedProvider,
)
from ragforge.adapters.fusion import ReciprocalRankFusion
from ragforge.adapters.lexical import BM25LexicalIndex
from ragforge.adapters.llm import GroqLLMProvider, OpenAILLMProvider
from ragforge.adapters.rerankers import DeterministicReranker
from ragforge.adapters.vector_stores import QdrantVectorStore
from ragforge.core.config import Settings, get_settings
from ragforge.domain.enums import SearchStrategy
from ragforge.ports.embeddings import BaseEmbeddingProvider
from ragforge.ports.fusion import BaseScoreFusion
from ragforge.ports.lexical import BaseLexicalIndex
from ragforge.ports.llm import BaseLLMProvider
from ragforge.ports.reranker import BaseReranker
from ragforge.ports.vector_store import BaseVectorStore
from ragforge.services.indexing import IndexingService
from ragforge.services.llm_router import LLMRouter
from ragforge.services.rag import RAGGenerationService
from ragforge.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# Shared lexical index instance shared across IndexingService and RetrievalService
_shared_lexical_index: BM25LexicalIndex | None = None


def reset_dependencies() -> None:
    """Reset shared dependency singletons (useful for test isolation)."""
    global _shared_lexical_index
    _shared_lexical_index = None


def get_lexical_index(settings: Settings | None = None) -> BaseLexicalIndex | None:
    """Get the shared BM25LexicalIndex instance if hybrid retrieval is enabled.

    Returns None if hybrid retrieval is disabled in configuration.
    When enabled, ensures the exact same lexical index instance is shared between
    IndexingService and RetrievalService.
    """
    global _shared_lexical_index
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    if not app_settings.hybrid_enabled:
        return None

    if _shared_lexical_index is None:
        logger.info(
            "Initializing shared BM25LexicalIndex instance for hybrid retrieval (index_file=%s).",
            app_settings.lexical_index_file,
        )
        _shared_lexical_index = BM25LexicalIndex(index_file=app_settings.lexical_index_file)
    return _shared_lexical_index


def get_fusion(settings: Settings | None = None) -> BaseScoreFusion | None:
    """Get the configured score fusion adapter if hybrid retrieval is enabled.

    Returns None if hybrid retrieval is disabled in configuration.
    """
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    if not app_settings.hybrid_enabled:
        return None

    if app_settings.fusion_strategy.lower() == "rrf":
        return ReciprocalRankFusion(
            k=app_settings.rrf_k,
            dense_weight=app_settings.rrf_dense_weight,
            lexical_weight=app_settings.rrf_lexical_weight,
        )

    raise ValueError(f"Unsupported fusion strategy: {app_settings.fusion_strategy}")


def get_embedding_provider(settings: Settings | None = None) -> BaseEmbeddingProvider:
    """Instantiate the configured embedding provider adapter."""
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    if app_settings.embedding_provider == "fastembed":
        return FastEmbedProvider(
            model_name=app_settings.embedding_model,
            cache_dir=app_settings.embedding_cache_dir,
        )
    return DeterministicEmbeddingProvider(dimension=app_settings.qdrant_vector_dimension)


def get_vector_store(settings: Settings | None = None) -> BaseVectorStore:
    """Instantiate the configured vector store adapter."""
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    return QdrantVectorStore(
        url=app_settings.qdrant_url,
        api_key=app_settings.qdrant_api_key,
        collection_name=app_settings.qdrant_collection,
        dimension=app_settings.qdrant_vector_dimension,
    )


def get_retrieval_service(
    settings: Annotated[Settings | None, Depends(get_settings)] = None,
) -> RetrievalService:
    """Dependency provider instantiating RetrievalService with wired ports."""
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    embedding_provider = get_embedding_provider(app_settings)
    vector_store = get_vector_store(app_settings)
    lexical_index = get_lexical_index(app_settings)
    fusion = get_fusion(app_settings)

    return RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        lexical_index=lexical_index,
        fusion=fusion,
        default_search_strategy=SearchStrategy.DENSE,
    )


def get_indexing_service(
    settings: Annotated[Settings | None, Depends(get_settings)] = None,
) -> IndexingService:
    """Dependency provider instantiating IndexingService with wired ports."""
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    embedding_provider = get_embedding_provider(app_settings)
    vector_store = get_vector_store(app_settings)
    lexical_index = get_lexical_index(app_settings)

    return IndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        batch_size=app_settings.indexing_batch_size,
        lexical_index=lexical_index,
    )


def get_primary_llm_provider(settings: Settings | None = None) -> BaseLLMProvider:
    """Instantiate primary LLM provider adapter."""
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    if app_settings.llm_provider == "openai":
        return OpenAILLMProvider(
            api_key=app_settings.openai_api_key,
            model=app_settings.llm_model,
            temperature=app_settings.llm_temperature,
            max_tokens=app_settings.llm_max_tokens,
            timeout=app_settings.llm_timeout,
        )
    return GroqLLMProvider(
        api_key=app_settings.groq_api_key,
        model=app_settings.llm_model,
        temperature=app_settings.llm_temperature,
        max_tokens=app_settings.llm_max_tokens,
        timeout=app_settings.llm_timeout,
    )


def get_fallback_llm_provider(settings: Settings | None = None) -> BaseLLMProvider | None:
    """Instantiate fallback LLM provider adapter if configured."""
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    if app_settings.llm_fallback_provider == "openai" and app_settings.openai_api_key:
        return OpenAILLMProvider(
            api_key=app_settings.openai_api_key,
            model=app_settings.llm_fallback_model,
            temperature=app_settings.llm_temperature,
            max_tokens=app_settings.llm_max_tokens,
            timeout=app_settings.llm_timeout,
        )
    if app_settings.llm_fallback_provider == "groq" and app_settings.groq_api_key:
        return GroqLLMProvider(
            api_key=app_settings.groq_api_key,
            model=app_settings.llm_fallback_model,
            temperature=app_settings.llm_temperature,
            max_tokens=app_settings.llm_max_tokens,
            timeout=app_settings.llm_timeout,
        )
    return None


def get_reranker(settings: Settings | None = None) -> BaseReranker | None:
    """Instantiate the configured reranker adapter if reranking is enabled."""
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    if not app_settings.reranking_enabled:
        return None

    if app_settings.reranker_provider == "deterministic":
        return DeterministicReranker()

    logger.warning(
        "Unknown reranker provider '%s'; falling back to deterministic",
        app_settings.reranker_provider,
    )
    return DeterministicReranker()


def get_rag_service(
    settings: Annotated[Settings | None, Depends(get_settings)] = None,
) -> RAGGenerationService:
    """Dependency provider instantiating the default RAG generation service."""
    app_settings = settings if isinstance(settings, Settings) else get_settings()
    retrieval_service = get_retrieval_service(app_settings)
    primary_provider = get_primary_llm_provider(app_settings)
    fallback_provider = get_fallback_llm_provider(app_settings)
    reranker = get_reranker(app_settings)

    llm_router = LLMRouter(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )
    return RAGGenerationService(
        retrieval_service=retrieval_service,
        llm_router=llm_router,
        reranker=reranker,
        reranking_enabled=app_settings.reranking_enabled,
        candidate_k=app_settings.rerank_candidate_k,
        default_top_k=app_settings.rerank_top_k if app_settings.reranking_enabled else 5,
        default_temperature=app_settings.llm_temperature,
        default_max_tokens=app_settings.llm_max_tokens,
        hybrid_candidate_k=app_settings.hybrid_candidate_k,
    )


__all__ = [
    "get_embedding_provider",
    "get_fallback_llm_provider",
    "get_fusion",
    "get_indexing_service",
    "get_lexical_index",
    "get_primary_llm_provider",
    "get_rag_service",
    "get_reranker",
    "get_retrieval_service",
    "get_vector_store",
    "reset_dependencies",
]
