"""Application dependency providers and wiring for FastAPI and domain services."""

import logging
from typing import Annotated, Any, cast

from fastapi import Depends, Request

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

# Shared lexical index for standalone usage across services
_shared_lexical_index: BM25LexicalIndex | None = None


def reset_dependencies() -> None:
    """Reset shared dependency singletons (useful for test isolation)."""
    global _shared_lexical_index
    _shared_lexical_index = None


def _register_closable(request: Request | None, resource: Any) -> None:
    """Register a closable adapter onto request.app.state for shutdown cleanup."""
    if request is None:
        return
    closables = getattr(request.app.state, "_closables", None)
    if closables is None:
        closables = []
        request.app.state._closables = closables
    if resource not in closables and hasattr(resource, "close"):
        closables.append(resource)


def _resolve_settings(settings: Settings | None, request: Request | None) -> Settings:
    """Resolve Settings instance preferring explicit argument, then app.state, then global."""
    if isinstance(settings, Settings):
        return settings
    if request is not None:
        app_settings = getattr(request.app.state, "settings", None)
        if isinstance(app_settings, Settings):
            return app_settings
    return get_settings()


def get_lexical_index(
    settings: Settings | None = None,
    request: Request = None,  # type: ignore[assignment]
) -> BaseLexicalIndex | None:
    """Get application-scoped or shared BM25LexicalIndex if hybrid retrieval is enabled.

    Returns None if hybrid retrieval is disabled in configuration.
    When enabled, ensures the lexical index instance is shared between
    IndexingService and RetrievalService within the application scope.
    """
    if request is not None and get_lexical_index in request.app.dependency_overrides:
        return cast(BaseLexicalIndex | None, request.app.dependency_overrides[get_lexical_index]())

    global _shared_lexical_index
    app_settings = _resolve_settings(settings, request)
    if not app_settings.hybrid_enabled:
        return None

    if request is not None:
        lexical = getattr(request.app.state, "lexical_index", None)
        if lexical is None:
            logger.info(
                "Initializing app-scoped BM25LexicalIndex for hybrid retrieval (index_file=%s).",
                app_settings.lexical_index_file,
            )
            lexical = BM25LexicalIndex(index_file=app_settings.lexical_index_file)
            request.app.state.lexical_index = lexical
        return cast(BaseLexicalIndex, lexical)

    if _shared_lexical_index is None:
        logger.info(
            "Initializing shared BM25LexicalIndex instance for hybrid retrieval (index_file=%s).",
            app_settings.lexical_index_file,
        )
        _shared_lexical_index = BM25LexicalIndex(index_file=app_settings.lexical_index_file)
    return _shared_lexical_index


def get_fusion(
    settings: Settings | None = None,
    request: Request = None,  # type: ignore[assignment]
) -> BaseScoreFusion | None:
    """Get the configured score fusion adapter if hybrid retrieval is enabled.

    Returns None if hybrid retrieval is disabled in configuration.
    """
    if request is not None and get_fusion in request.app.dependency_overrides:
        return cast(BaseScoreFusion | None, request.app.dependency_overrides[get_fusion]())

    app_settings = _resolve_settings(settings, request)
    if not app_settings.hybrid_enabled:
        return None

    if request is not None:
        fusion = getattr(request.app.state, "fusion", None)
        if fusion is not None:
            return cast(BaseScoreFusion, fusion)

    if app_settings.fusion_strategy.lower() == "rrf":
        fusion = ReciprocalRankFusion(
            k=app_settings.rrf_k,
            dense_weight=app_settings.rrf_dense_weight,
            lexical_weight=app_settings.rrf_lexical_weight,
        )
    else:
        raise ValueError(f"Unsupported fusion strategy: {app_settings.fusion_strategy}")

    if request is not None:
        request.app.state.fusion = fusion
    return fusion


def get_embedding_provider(
    settings: Settings | None = None,
    request: Request = None,  # type: ignore[assignment]
) -> BaseEmbeddingProvider:
    """Instantiate or retrieve the application-scoped embedding provider adapter."""
    if request is not None and get_embedding_provider in request.app.dependency_overrides:
        return cast(
            BaseEmbeddingProvider, request.app.dependency_overrides[get_embedding_provider]()
        )

    app_settings = _resolve_settings(settings, request)
    if request is not None:
        provider = getattr(request.app.state, "embedding_provider", None)
        if provider is not None:
            return cast(BaseEmbeddingProvider, provider)

    if app_settings.embedding_provider == "fastembed":
        provider = FastEmbedProvider(
            model_name=app_settings.embedding_model,
            cache_dir=app_settings.embedding_cache_dir,
        )
    else:
        provider = DeterministicEmbeddingProvider(dimension=app_settings.qdrant_vector_dimension)

    if request is not None:
        request.app.state.embedding_provider = provider
    return provider


def get_vector_store(
    settings: Settings | None = None,
    request: Request = None,  # type: ignore[assignment]
) -> BaseVectorStore:
    """Instantiate or retrieve the application-scoped vector store adapter."""
    if request is not None and get_vector_store in request.app.dependency_overrides:
        return cast(BaseVectorStore, request.app.dependency_overrides[get_vector_store]())

    app_settings = _resolve_settings(settings, request)
    if request is not None:
        store = getattr(request.app.state, "vector_store", None)
        if store is not None:
            return cast(BaseVectorStore, store)

    store = QdrantVectorStore(
        url=app_settings.qdrant_url,
        api_key=app_settings.qdrant_api_key,
        collection_name=app_settings.qdrant_collection,
        dimension=app_settings.qdrant_vector_dimension,
    )
    if request is not None:
        request.app.state.vector_store = store
        _register_closable(request, store)
    return store


def get_primary_llm_provider(
    settings: Settings | None = None,
    request: Request = None,  # type: ignore[assignment]
) -> BaseLLMProvider:
    """Instantiate primary LLM provider adapter."""
    if request is not None and get_primary_llm_provider in request.app.dependency_overrides:
        return cast(BaseLLMProvider, request.app.dependency_overrides[get_primary_llm_provider]())

    app_settings = _resolve_settings(settings, request)
    if request is not None:
        provider = getattr(request.app.state, "primary_llm_provider", None)
        if provider is not None:
            return cast(BaseLLMProvider, provider)

    if app_settings.llm_provider == "openai":
        provider = OpenAILLMProvider(
            api_key=app_settings.openai_api_key,
            model=app_settings.llm_model,
            temperature=app_settings.llm_temperature,
            max_tokens=app_settings.llm_max_tokens,
            timeout=app_settings.llm_timeout,
        )
    else:
        provider = GroqLLMProvider(
            api_key=app_settings.groq_api_key,
            model=app_settings.llm_model,
            temperature=app_settings.llm_temperature,
            max_tokens=app_settings.llm_max_tokens,
            timeout=app_settings.llm_timeout,
        )

    if request is not None:
        request.app.state.primary_llm_provider = provider
    return provider


def get_fallback_llm_provider(
    settings: Settings | None = None,
    request: Request = None,  # type: ignore[assignment]
) -> BaseLLMProvider | None:
    """Instantiate fallback LLM provider adapter if configured."""
    if request is not None and get_fallback_llm_provider in request.app.dependency_overrides:
        return cast(
            BaseLLMProvider | None, request.app.dependency_overrides[get_fallback_llm_provider]()
        )

    app_settings = _resolve_settings(settings, request)
    if request is not None and hasattr(request.app.state, "fallback_llm_provider"):
        return cast(BaseLLMProvider | None, request.app.state.fallback_llm_provider)

    provider: BaseLLMProvider | None = None
    if app_settings.llm_fallback_provider == "openai" and app_settings.openai_api_key:
        provider = OpenAILLMProvider(
            api_key=app_settings.openai_api_key,
            model=app_settings.llm_fallback_model,
            temperature=app_settings.llm_temperature,
            max_tokens=app_settings.llm_max_tokens,
            timeout=app_settings.llm_timeout,
        )
    elif app_settings.llm_fallback_provider == "groq" and app_settings.groq_api_key:
        provider = GroqLLMProvider(
            api_key=app_settings.groq_api_key,
            model=app_settings.llm_fallback_model,
            temperature=app_settings.llm_temperature,
            max_tokens=app_settings.llm_max_tokens,
            timeout=app_settings.llm_timeout,
        )

    if request is not None:
        request.app.state.fallback_llm_provider = provider
    return provider


def get_reranker(
    settings: Settings | None = None,
    request: Request = None,  # type: ignore[assignment]
) -> BaseReranker | None:
    """Instantiate the configured reranker adapter if reranking is enabled."""
    if request is not None and get_reranker in request.app.dependency_overrides:
        return cast(BaseReranker | None, request.app.dependency_overrides[get_reranker]())

    app_settings = _resolve_settings(settings, request)
    if not app_settings.reranking_enabled:
        return None

    if request is not None:
        reranker = getattr(request.app.state, "reranker", None)
        if reranker is not None:
            return cast(BaseReranker, reranker)

    if app_settings.reranker_provider == "deterministic":
        reranker = DeterministicReranker()
    else:
        logger.warning(
            "Unknown reranker provider '%s'; falling back to deterministic",
            app_settings.reranker_provider,
        )
        reranker = DeterministicReranker()

    if request is not None:
        request.app.state.reranker = reranker
    return reranker


def get_retrieval_service(
    settings: Annotated[Settings | None, Depends(get_settings)] = None,
    request: Request = None,  # type: ignore[assignment]
) -> RetrievalService:
    """Dependency provider instantiating RetrievalService with wired ports."""
    if request is not None and get_retrieval_service in request.app.dependency_overrides:
        return cast(RetrievalService, request.app.dependency_overrides[get_retrieval_service]())

    app_settings = _resolve_settings(settings, request)
    if request is not None:
        cached_service = getattr(request.app.state, "retrieval_service", None)
        if cached_service is not None:
            return cast(RetrievalService, cached_service)

    embedding_provider = get_embedding_provider(app_settings, request=request)
    vector_store = get_vector_store(app_settings, request=request)
    lexical_index = get_lexical_index(app_settings, request=request)
    fusion = get_fusion(app_settings, request=request)

    service = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        lexical_index=lexical_index,
        fusion=fusion,
        default_search_strategy=SearchStrategy.DENSE,
    )
    if request is not None:
        request.app.state.retrieval_service = service
    return service


def get_indexing_service(
    settings: Annotated[Settings | None, Depends(get_settings)] = None,
    request: Request = None,  # type: ignore[assignment]
) -> IndexingService:
    """Dependency provider instantiating IndexingService with wired ports."""
    if request is not None and get_indexing_service in request.app.dependency_overrides:
        return cast(IndexingService, request.app.dependency_overrides[get_indexing_service]())

    app_settings = _resolve_settings(settings, request)
    if request is not None:
        cached_service = getattr(request.app.state, "indexing_service", None)
        if cached_service is not None:
            return cast(IndexingService, cached_service)

    embedding_provider = get_embedding_provider(app_settings, request=request)
    vector_store = get_vector_store(app_settings, request=request)
    lexical_index = get_lexical_index(app_settings, request=request)

    service = IndexingService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        batch_size=app_settings.indexing_batch_size,
        lexical_index=lexical_index,
    )
    if request is not None:
        request.app.state.indexing_service = service
    return service


def get_rag_service(
    settings: Annotated[Settings | None, Depends(get_settings)] = None,
    request: Request = None,  # type: ignore[assignment]
) -> RAGGenerationService:
    """Dependency provider instantiating the default RAG generation service."""
    if request is not None and get_rag_service in request.app.dependency_overrides:
        return cast(RAGGenerationService, request.app.dependency_overrides[get_rag_service]())

    app_settings = _resolve_settings(settings, request)
    if request is not None:
        cached_rag = getattr(request.app.state, "rag_service", None)
        if cached_rag is not None:
            return cast(RAGGenerationService, cached_rag)

    retrieval_service = get_retrieval_service(app_settings, request=request)
    primary_provider = get_primary_llm_provider(app_settings, request=request)
    fallback_provider = get_fallback_llm_provider(app_settings, request=request)
    reranker = get_reranker(app_settings, request=request)

    llm_router = LLMRouter(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
    )
    service = RAGGenerationService(
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
    if request is not None:
        request.app.state.rag_service = service
    return service


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
