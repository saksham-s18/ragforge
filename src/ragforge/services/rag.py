import logging
from typing import Any

from ragforge.domain.enums import SearchStrategy
from ragforge.domain.models import (
    GenerationRequest,
    RAGResponse,
    SourceReference,
)
from ragforge.ports.reranker import BaseReranker
from ragforge.services.llm_router import LLMRouter
from ragforge.services.prompt import PromptBuilder
from ragforge.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class RAGGenerationService:
    """Production RAG orchestration service coordinating retrieval, context construction,
    grounded prompt assembly, multi-provider LLM routing, and source attribution.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_router: LLMRouter,
        prompt_builder: PromptBuilder | None = None,
        reranker: BaseReranker | None = None,
        reranking_enabled: bool | None = None,
        candidate_k: int | None = None,
        default_top_k: int = 5,
        default_temperature: float = 0.0,
        default_max_tokens: int | None = 1024,
        hybrid_candidate_k: int = 20,
    ) -> None:
        """Initialize the RAG generation service.

        Args:
            retrieval_service: Orchestrator for dense vector search and chunk retrieval.
            llm_router: Routing and fallback manager for LLM provider adapters.
            prompt_builder: Deterministic prompt and context construction helper.
            reranker: Optional second-stage reranker port for candidate refinement.
            reranking_enabled: Optional explicit boolean flag controlling whether reranking
                is active (defaults to True if reranker is provided, else False).
            candidate_k: Default candidate retrieval depth when reranking is active.
            default_top_k: Default number of final context chunks to select.
            default_temperature: Default sampling temperature for generation.
            default_max_tokens: Default maximum output tokens to produce.
            hybrid_candidate_k: Candidate pool size retrieved from hybrid channels.
        """
        self._retrieval_service = retrieval_service
        self._llm_router = llm_router
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._reranker = reranker
        if reranking_enabled is not None:
            self._reranking_enabled = reranking_enabled
        else:
            self._reranking_enabled = reranker is not None
        self._candidate_k = candidate_k
        self._default_top_k = default_top_k
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._hybrid_candidate_k = hybrid_candidate_k

    async def generate_answer(
        self,
        question: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        search_strategy: SearchStrategy | None = None,
    ) -> RAGResponse:
        """Execute end-to-end RAG answer synthesis for a user question.

        Args:
            question: User inquiry text.
            top_k: Optional number of context chunks to select (defaults to instance setting).
            candidate_k: Optional candidate retrieval count before reranking or slicing.
            filters: Optional metadata filters applied to vector store retrieval.
            score_threshold: Optional minimum cosine similarity score cutoff.
            temperature: Sampling temperature override.
            max_tokens: Maximum completion tokens override.
            model: Optional model identifier override passed to the provider router.
            search_strategy: Optional search strategy (dense, sparse, hybrid).

        Returns:
            Structured RAGResponse containing the synthesized answer, application-managed
            source citations, provider metadata, and token usage information.

        Raises:
            ValueError: If the question is empty or consists solely of whitespace.
            LLMError: When all eligible LLM providers fail to generate an answer.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty or whitespace.")

        effective_top_k = top_k if top_k is not None else self._default_top_k
        effective_temp = temperature if temperature is not None else self._default_temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self._default_max_tokens

        use_reranking = self._reranking_enabled and (self._reranker is not None)
        is_hybrid = search_strategy == SearchStrategy.HYBRID

        # 1. Determine candidate retrieval depth
        if is_hybrid and use_reranking:
            hybrid_target = candidate_k if candidate_k is not None else self._hybrid_candidate_k
            rerank_target = (
                candidate_k
                if candidate_k is not None
                else (
                    self._candidate_k
                    if self._candidate_k is not None
                    else max(effective_top_k * 4, 20)
                )
            )
            retrieval_k = max(effective_top_k, hybrid_target, rerank_target)
        elif is_hybrid:
            hybrid_target = candidate_k if candidate_k is not None else self._hybrid_candidate_k
            retrieval_k = max(effective_top_k, hybrid_target)
        elif use_reranking:
            rerank_target = (
                candidate_k
                if candidate_k is not None
                else (
                    self._candidate_k
                    if self._candidate_k is not None
                    else max(effective_top_k * 4, 20)
                )
            )
            retrieval_k = max(effective_top_k, rerank_target)
        else:
            retrieval_k = effective_top_k

        logger.info(
            "Executing RAG generation for question: '%s' (retrieval_k=%d, top_k=%d, reranking=%s)",
            cleaned_question[:60],
            retrieval_k,
            effective_top_k,
            use_reranking,
        )

        # 2. Retrieve candidates via RetrievalService
        if search_strategy is not None:
            retrieved_items = await self._retrieval_service.retrieve(
                query=cleaned_question,
                top_k=retrieval_k,
                filters=filters,
                score_threshold=score_threshold,
                search_strategy=search_strategy,
            )
        else:
            retrieved_items = await self._retrieval_service.retrieve(
                query=cleaned_question,
                top_k=retrieval_k,
                filters=filters,
                score_threshold=score_threshold,
            )

        # 3. Downstream Reranker (Stage 10) or candidate slicing
        if use_reranking:
            assert self._reranker is not None
            logger.debug(
                "Passing %d candidates to reranker (target top_k=%d)",
                len(retrieved_items),
                effective_top_k,
            )
            reranked_chunks = await self._reranker.rerank(
                query=cleaned_question,
                chunks=retrieved_items,
                top_k=effective_top_k,
            )
            final_chunks = reranked_chunks[:effective_top_k]
            logger.debug("Selected %d top chunk(s) after reranking", len(final_chunks))
        else:
            final_chunks = retrieved_items[:effective_top_k]
            logger.debug("Selected %d context chunk(s) without reranking", len(final_chunks))

        # 4. Build application-managed source references (preserving provenance and scores)
        sources: list[SourceReference] = []
        for item in final_chunks:
            chunk = item.chunk
            extra = chunk.metadata.extra or {}
            snippet = (
                chunk.content[:250].strip() + "..."
                if len(chunk.content) > 250
                else chunk.content.strip()
            )
            sources.append(
                SourceReference(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    source_path=extra.get("file_path") or extra.get("source"),
                    document_title=extra.get("document_title") or extra.get("file_name"),
                    section_header=chunk.metadata.section_header,
                    page_number=chunk.metadata.page_number,
                    start_char_idx=chunk.metadata.start_char_idx,
                    end_char_idx=chunk.metadata.end_char_idx,
                    score=round(item.score, 4),
                    rerank_score=(
                        round(item.rerank_score, 4) if item.rerank_score is not None else None
                    ),
                    content_snippet=snippet,
                )
            )

        # 5. Assemble grounded prompt and deterministic context
        system_prompt = self._prompt_builder.build_system_prompt()
        user_prompt = self._prompt_builder.build_user_prompt(
            question=cleaned_question,
            chunks=final_chunks,
        )

        # 6. Formulate generation request
        gen_request = GenerationRequest(
            prompt=user_prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=effective_temp,
            max_tokens=effective_max_tokens,
            metadata={
                "retrieved_chunks": len(final_chunks),
                "candidate_chunks": len(retrieved_items),
            },
        )

        # 7. Delegate to LLM provider router
        llm_response = await self._llm_router.generate(gen_request)

        # 8. Return comprehensive RAGResponse
        response_metadata: dict[str, Any] = {
            "retrieved_count": len(retrieved_items),
            "context_count": len(final_chunks),
            "top_k": effective_top_k,
            "reranked": use_reranking,
        }
        if use_reranking:
            response_metadata["candidate_count"] = len(retrieved_items)

        return RAGResponse(
            question=cleaned_question,
            answer=llm_response.text,
            sources=sources,
            provider=llm_response.provider,
            model=llm_response.model,
            usage=llm_response.usage,
            metadata=response_metadata,
        )
