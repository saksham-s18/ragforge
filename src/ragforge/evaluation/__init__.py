"""Evaluation package for RAGForge retrieval benchmarking and quality assessment."""

from ragforge.evaluation.dataset import (
    load_evaluation_dataset,
    save_evaluation_dataset,
    validate_evaluation_dataset,
)
from ragforge.evaluation.matcher import (
    evaluate_case_relevance,
    is_chunk_match,
    is_chunk_relevant,
    is_source_match,
)
from ragforge.evaluation.metrics import (
    mean_precision_at_k,
    mean_recall_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    precision_at_k_binary,
    recall_at_k,
    recall_at_k_binary,
    reciprocal_rank,
    reciprocal_rank_binary,
)
from ragforge.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "evaluate_case_relevance",
    "is_chunk_match",
    "is_chunk_relevant",
    "is_source_match",
    "load_evaluation_dataset",
    "mean_precision_at_k",
    "mean_recall_at_k",
    "mean_reciprocal_rank",
    "precision_at_k",
    "precision_at_k_binary",
    "recall_at_k",
    "recall_at_k_binary",
    "reciprocal_rank",
    "reciprocal_rank_binary",
    "save_evaluation_dataset",
    "validate_evaluation_dataset",
]
