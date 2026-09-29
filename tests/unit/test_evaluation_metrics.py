"""Unit tests for IR retrieval metrics: Recall@K, Precision@K, and MRR."""

import pytest

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

# --------------------------------------------------------------------------
# Precision@K Tests
# --------------------------------------------------------------------------


def test_precision_at_k_basic() -> None:
    """Verify Precision@K computation on typical rankings."""
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3"}

    assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)
    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 4)
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.4)


def test_precision_at_k_k_larger_than_results() -> None:
    """Verify Precision@K when K is larger than available retrieved results."""
    retrieved = ["doc1", "doc2"]
    relevant = {"doc1"}

    # With only 2 results retrieved, denominator remains K=5
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.2)
    assert precision_at_k(retrieved, relevant, k=10) == pytest.approx(0.1)


def test_precision_at_k_empty_results() -> None:
    """Verify Precision@K returns 0.0 when retrieved list is empty."""
    assert precision_at_k([], {"doc1"}, k=5) == 0.0


def test_precision_at_k_no_relevant_items() -> None:
    """Verify Precision@K returns 0.0 when relevant set is empty."""
    assert precision_at_k(["doc1", "doc2"], set(), k=5) == 0.0


def test_precision_at_k_zero_or_negative_k() -> None:
    """Verify Precision@K returns 0.0 for k <= 0."""
    assert precision_at_k(["doc1"], {"doc1"}, k=0) == 0.0
    assert precision_at_k(["doc1"], {"doc1"}, k=-3) == 0.0


# --------------------------------------------------------------------------
# Recall@K Tests
# --------------------------------------------------------------------------


def test_recall_at_k_basic() -> None:
    """Verify Recall@K computation on standard rankings."""
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc1", "doc3", "doc5"}  # 3 total relevant items

    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)


def test_recall_at_k_k_larger_than_results() -> None:
    """Verify Recall@K when K is larger than retrieved results."""
    retrieved = ["doc1", "doc2"]
    relevant = {"doc1"}

    # 1 of 1 relevant item retrieved
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)
    assert recall_at_k(retrieved, relevant, k=100) == pytest.approx(1.0)


def test_recall_at_k_empty_results() -> None:
    """Verify Recall@K returns 0.0 when retrieved list is empty."""
    assert recall_at_k([], {"doc1"}, k=5) == 0.0


def test_recall_at_k_no_relevant_items() -> None:
    """Verify Recall@K handles empty ground truth gracefully without ZeroDivisionError."""
    assert recall_at_k(["doc1", "doc2"], set(), k=5) == 0.0
    assert recall_at_k(["doc1", "doc2"], {"doc1"}, k=5, total_relevant=0) == 0.0


def test_recall_at_k_zero_or_negative_k() -> None:
    """Verify Recall@K returns 0.0 for k <= 0."""
    assert recall_at_k(["doc1"], {"doc1"}, k=0) == 0.0
    assert recall_at_k(["doc1"], {"doc1"}, k=-1) == 0.0


def test_recall_at_k_duplicate_retrieved_items() -> None:
    """Verify duplicate occurrences of the same relevant item do not inflate recall."""
    retrieved = ["doc1", "doc1", "doc2"]
    relevant = {"doc1", "doc3"}  # 2 total relevant items

    # doc1 is retrieved twice in top 2, but only represents 1 unique hit
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Reciprocal Rank (RR) and MRR Tests
# --------------------------------------------------------------------------


def test_reciprocal_rank_first_rank() -> None:
    """Verify RR is 1.0 when relevant item is at rank 1."""
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = {"doc1"}
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1.0)


def test_reciprocal_rank_subsequent_rank() -> None:
    """Verify RR calculation for relevant item at rank > 1."""
    retrieved = ["docX", "docY", "doc1", "doc2"]
    relevant = {"doc1"}
    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found() -> None:
    """Verify RR is 0.0 when no relevant items are retrieved."""
    retrieved = ["docX", "docY", "docZ"]
    relevant = {"docA", "docB"}
    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_reciprocal_rank_empty_results() -> None:
    """Verify RR is 0.0 when retrieved list is empty."""
    assert reciprocal_rank([], {"doc1"}) == 0.0


def test_reciprocal_rank_cutoff_k() -> None:
    """Verify RR respects ranking cutoff threshold K."""
    retrieved = ["docX", "docY", "docZ", "doc1"]
    relevant = {"doc1"}

    # doc1 is at rank 4
    assert reciprocal_rank(retrieved, relevant, k=3) == 0.0
    assert reciprocal_rank(retrieved, relevant, k=4) == pytest.approx(0.25)
    assert reciprocal_rank(retrieved, relevant, k=0) == 0.0


def test_mean_reciprocal_rank_aggregation() -> None:
    """Verify MRR aggregation across multiple queries."""
    rrs = [1.0, 0.5, 0.0, 0.25]
    expected_mrr = (1.0 + 0.5 + 0.0 + 0.25) / 4.0
    assert mean_reciprocal_rank(rrs) == pytest.approx(expected_mrr)
    assert mean_reciprocal_rank([]) == 0.0


def test_mean_precision_and_recall_aggregation() -> None:
    """Verify macro-average precision and recall calculations."""
    vals = [1.0, 0.5, 0.0]
    expected = 1.5 / 3.0
    assert mean_precision_at_k(vals) == pytest.approx(expected)
    assert mean_recall_at_k(vals) == pytest.approx(expected)
    assert mean_precision_at_k([]) == 0.0
    assert mean_recall_at_k([]) == 0.0


# --------------------------------------------------------------------------
# Binary Sequence Variants Tests
# --------------------------------------------------------------------------


def test_binary_metrics_variants() -> None:
    """Verify binary relevance metric functions."""
    relevance = [True, False, True, False, False]

    assert precision_at_k_binary(relevance, k=1) == pytest.approx(1.0)
    assert precision_at_k_binary(relevance, k=2) == pytest.approx(0.5)
    assert precision_at_k_binary(relevance, k=5) == pytest.approx(0.4)
    assert precision_at_k_binary(relevance, k=0) == 0.0

    assert recall_at_k_binary(relevance, total_relevant=2, k=1) == pytest.approx(0.5)
    assert recall_at_k_binary(relevance, total_relevant=2, k=3) == pytest.approx(1.0)
    assert recall_at_k_binary(relevance, total_relevant=0, k=5) == 0.0

    assert reciprocal_rank_binary(relevance) == pytest.approx(1.0)
    assert reciprocal_rank_binary([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank_binary([False, False, False]) == 0.0
    assert reciprocal_rank_binary([False, True], k=1) == 0.0
