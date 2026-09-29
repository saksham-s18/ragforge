"""Retrieval evaluation orchestration service."""

from ragforge.evaluation.runner import EvaluationRunner

# Alias EvaluationRunner as RetrievalEvaluationService for service architecture symmetry
RetrievalEvaluationService = EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "RetrievalEvaluationService",
]
