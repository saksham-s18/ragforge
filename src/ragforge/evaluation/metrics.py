"""Information Retrieval (IR) evaluation metrics for RAG retrieval benchmarking.

Provides mathematically rigorous, pure-function implementations of:
- Precision@K
- Recall@K
- Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)
- Aggregate mean calculations across query distributions

All functions safely handle edge cases such as empty result sets, missing relevant targets,
K greater than available results, and non-positive K cutoffs without throwing exceptions.
"""

from collections.abc import Sequence


def precision_at_k[T](
    retrieved: Sequence[T],
    relevant: set[T] | Sequence[T],
    k: int,
) -> float:
    """Compute Precision@K for a retrieved ranking and set of relevant target items.

    Precision@K measures the proportion of retrieved documents in the top-K ranking
    that are relevant:

        Precision@K = (number of relevant items in top K) / K

    Args:
        retrieved: Ordered sequence of retrieved item identifiers (rank 1 to N).
        relevant: Collection or set of ground-truth relevant item identifiers.
        k: Ranking cutoff threshold.

    Returns:
        Precision@K value in the range [0.0, 1.0]. Returns 0.0 if k <= 0, retrieved
        is empty, or no relevant items exist.
    """
    if k <= 0 or not retrieved:
        return 0.0

    rel_set = set(relevant)
    if not rel_set:
        return 0.0

    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in rel_set)
    return hits / float(k)


def recall_at_k[T](
    retrieved: Sequence[T],
    relevant: set[T] | Sequence[T],
    k: int,
    total_relevant: int | None = None,
) -> float:
    """Compute Recall@K for a retrieved ranking and set of relevant target items.

    Recall@K measures the proportion of all relevant documents that appear in the
    top-K retrieved results:

        Recall@K = (number of unique relevant items in top K) / (total relevant items)

    Args:
        retrieved: Ordered sequence of retrieved item identifiers (rank 1 to N).
        relevant: Collection or set of ground-truth relevant item identifiers.
        k: Ranking cutoff threshold.
        total_relevant: Optional ground-truth denominator override. If omitted,
            len(set(relevant)) is used.

    Returns:
        Recall@K value in the range [0.0, 1.0]. Returns 0.0 if k <= 0, retrieved
        is empty, or total expected relevant items is 0.
    """
    if k <= 0 or not retrieved:
        return 0.0

    rel_set = set(relevant)
    denominator = total_relevant if total_relevant is not None else len(rel_set)
    if denominator <= 0:
        return 0.0

    top_k = retrieved[:k]
    # Unique relevant items retrieved in top K
    hits = len(set(top_k) & rel_set)
    return min(1.0, hits / float(denominator))


def reciprocal_rank[T](
    retrieved: Sequence[T],
    relevant: set[T] | Sequence[T],
    k: int | None = None,
) -> float:
    """Compute Reciprocal Rank (RR) for a retrieved ranking and relevant targets.

    RR evaluates the rank position of the FIRST relevant item in the ranking:

        RR = 1 / rank_of_first_relevant_item

    If no relevant item is found within the retrieved list (or within top K if specified),
    the score is 0.0.

    Args:
        retrieved: Ordered sequence of retrieved item identifiers (rank 1 to N).
        relevant: Collection or set of ground-truth relevant item identifiers.
        k: Optional cutoff threshold. If specified, only ranks <= k are considered.

    Returns:
        Reciprocal Rank in the range [0.0, 1.0]. Returns 0.0 if no relevant items
        are found or if k <= 0.
    """
    if k is not None and k <= 0:
        return 0.0

    rel_set = set(relevant)
    if not rel_set or not retrieved:
        return 0.0

    candidates = retrieved[:k] if k is not None else retrieved
    for rank, item in enumerate(candidates, start=1):
        if item in rel_set:
            return 1.0 / float(rank)

    return 0.0


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    """Compute Mean Reciprocal Rank (MRR) across a collection of per-query RR values.

    Args:
        reciprocal_ranks: Sequence of individual query RR scores.

    Returns:
        Mean RR value in the range [0.0, 1.0]. Returns 0.0 if reciprocal_ranks is empty.
    """
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / float(len(reciprocal_ranks))


def mean_precision_at_k(precisions: Sequence[float]) -> float:
    """Compute Mean Precision@K across multiple evaluated queries."""
    if not precisions:
        return 0.0
    return sum(precisions) / float(len(precisions))


def mean_recall_at_k(recalls: Sequence[float]) -> float:
    """Compute Mean Recall@K across multiple evaluated queries."""
    if not recalls:
        return 0.0
    return sum(recalls) / float(len(recalls))


# -------------------------------------------------------------------------
# Binary relevance array variants (e.g. [True, False, False, True, ...])
# -------------------------------------------------------------------------


def precision_at_k_binary(relevance: Sequence[bool | int], k: int) -> float:
    """Compute Precision@K given a pre-computed binary relevance sequence.

    Args:
        relevance: Sequence where True/1 indicates the candidate at that rank is relevant.
        k: Cutoff rank.

    Returns:
        Precision@K score.
    """
    if k <= 0 or not relevance:
        return 0.0
    hits = sum(1 for r in relevance[:k] if r)
    return hits / float(k)


def recall_at_k_binary(
    relevance: Sequence[bool | int],
    total_relevant: int,
    k: int,
) -> float:
    """Compute Recall@K given a pre-computed binary relevance sequence.

    Args:
        relevance: Sequence where True/1 indicates the candidate at that rank is relevant.
        total_relevant: Total number of ground-truth relevant items.
        k: Cutoff rank.

    Returns:
        Recall@K score.
    """
    if k <= 0 or not relevance or total_relevant <= 0:
        return 0.0
    hits = sum(1 for r in relevance[:k] if r)
    return min(1.0, hits / float(total_relevant))


def reciprocal_rank_binary(
    relevance: Sequence[bool | int],
    k: int | None = None,
) -> float:
    """Compute Reciprocal Rank (RR) given a pre-computed binary relevance sequence.

    Args:
        relevance: Sequence where True/1 indicates the candidate at that rank is relevant.
        k: Optional cutoff rank.

    Returns:
        Reciprocal Rank score.
    """
    if k is not None and k <= 0:
        return 0.0
    candidates = relevance[:k] if k is not None else relevance
    for rank, r in enumerate(candidates, start=1):
        if r:
            return 1.0 / float(rank)
    return 0.0
