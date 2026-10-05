"""Evaluation runner orchestrating retrieval execution, relevance scoring, and metrics."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ragforge.domain.models import (
    CaseEvaluationResult,
    EvaluationCase,
    EvaluationDataset,
    RetrievalEvaluationReport,
)
from ragforge.evaluation.dataset import validate_evaluation_dataset
from ragforge.evaluation.matcher import evaluate_case_relevance
from ragforge.evaluation.metrics import (
    mean_precision_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
)

if TYPE_CHECKING:
    from ragforge.ports.reranker import BaseReranker
    from ragforge.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates end-to-end retrieval benchmarking across evaluation datasets.

    Pipeline:
        Dataset -> Runner -> RetrievalService -> VectorStore
        -> [Optional Reranker] -> Retrieved Results -> Metrics
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        k_values: Sequence[int] | None = None,
        reranker: BaseReranker | None = None,
        candidate_k: int | None = None,
    ) -> None:
        """Initialize the evaluation runner.

        Args:
            retrieval_service: Configured application RetrievalService port.
            k_values: Sequence of K cutoff values (defaults to [1, 3, 5]).
            reranker: Optional BaseReranker port to evaluate second-stage reranking.
            candidate_k: Optional candidate retrieval depth before reranking.

        Raises:
            ValueError: If k_values contains non-positive numbers or is empty.
        """
        self._retrieval_service = retrieval_service
        self._reranker = reranker
        self._candidate_k = candidate_k
        raw_k = list(k_values) if k_values is not None else [1, 3, 5]
        if not raw_k or any(k <= 0 for k in raw_k):
            raise ValueError(f"k_values must be non-empty positive integers, got {raw_k}")
        self.k_values = sorted(dict.fromkeys(raw_k))

    async def evaluate_case(
        self,
        case: EvaluationCase,
        top_k: int | None = None,
    ) -> CaseEvaluationResult:
        """Execute retrieval and compute relevance metrics for an individual evaluation case.

        Args:
            case: EvaluationCase defining the user question and expected targets.
            top_k: Optional maximum retrieval depth override (defaults to max(k_values)).

        Returns:
            CaseEvaluationResult containing individual case metrics and attribution.
        """
        effective_top_k = top_k if top_k is not None else max(self.k_values)
        if effective_top_k <= 0:
            raise ValueError(f"top_k must be positive, got {effective_top_k}")

        if self._reranker is not None:
            candidate_pool = (
                self._candidate_k
                if self._candidate_k is not None
                else max(max(self.k_values) * 2, 20)
            )
            candidate_depth = max(candidate_pool, effective_top_k)
            candidate_chunks = await self._retrieval_service.retrieve(
                query=case.question,
                top_k=candidate_depth,
            )
            retrieved_chunks = await self._reranker.rerank(
                query=case.question,
                chunks=candidate_chunks,
                top_k=effective_top_k,
            )
        else:
            retrieved_chunks = await self._retrieval_service.retrieve(
                query=case.question,
                top_k=effective_top_k,
            )

        return evaluate_case_relevance(
            retrieved_chunks=retrieved_chunks,
            case=case,
            k_values=self.k_values,
        )

    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        top_k: int | None = None,
    ) -> RetrievalEvaluationReport:
        """Run batch retrieval evaluation across all cases in an EvaluationDataset.

        Args:
            dataset: EvaluationDataset to evaluate.
            top_k: Optional maximum retrieval depth override.

        Returns:
            RetrievalEvaluationReport containing macro-averaged IR metrics and case results.
        """
        start_time = time.perf_counter()
        validate_evaluation_dataset(dataset)

        logger.info(
            "Starting retrieval evaluation for dataset '%s' with %d cases (K=%s)",
            dataset.name,
            len(dataset.cases),
            self.k_values,
        )

        effective_top_k = top_k if top_k is not None else max(self.k_values)
        case_results: list[CaseEvaluationResult] = []

        for case in dataset.cases:
            result = await self.evaluate_case(case, top_k=effective_top_k)
            case_results.append(result)

        # Macro-average metrics across all query cases
        mean_recall: dict[int, float] = {}
        mean_precision: dict[int, float] = {}

        for k in self.k_values:
            recalls = [res.recall_at_k.get(k, 0.0) for res in case_results]
            precisions = [res.precision_at_k.get(k, 0.0) for res in case_results]
            mean_recall[k] = mean_recall_at_k(recalls)
            mean_precision[k] = mean_precision_at_k(precisions)

        reciprocal_ranks = [res.reciprocal_rank for res in case_results]
        mrr = mean_reciprocal_rank(reciprocal_ranks)
        duration = time.perf_counter() - start_time

        logger.info(
            "Evaluation completed in %.4fs: MRR=%.4f, Recall@1=%.4f, Precision@1=%.4f",
            duration,
            mrr,
            mean_recall.get(1, 0.0),
            mean_precision.get(1, 0.0),
        )

        return RetrievalEvaluationReport(
            dataset_name=dataset.name,
            total_cases=len(case_results),
            k_values=self.k_values,
            mean_recall_at_k=mean_recall,
            mean_precision_at_k=mean_precision,
            mean_reciprocal_rank=mrr,
            duration_seconds=duration,
            case_results=case_results,
        )
