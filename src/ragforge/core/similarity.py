import math
from collections.abc import Sequence

from ragforge.domain.exceptions import VectorDimensionMismatchError


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Compute mathematically sound cosine similarity between two real-valued vectors.

    Calculates:
        cos(theta) = (A . B) / (||A|| * ||B||)

    Args:
        vector_a: Sequence of float values representing vector A.
        vector_b: Sequence of float values representing vector B.

    Returns:
        Cosine similarity score bounded strictly in [-1.0, 1.0]. Returns 0.0
        if either vector is empty or a zero-vector (has zero magnitude).

    Raises:
        VectorDimensionMismatchError: If vectors have different dimensions.
    """
    if len(vector_a) != len(vector_b):
        raise VectorDimensionMismatchError(
            f"Vector dimension mismatch in cosine similarity: {len(vector_a)} != {len(vector_b)}"
        )

    if not vector_a:
        return 0.0

    dot_product = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0

    for a, b in zip(vector_a, vector_b, strict=True):
        dot_product += a * b
        norm_a_sq += a * a
        norm_b_sq += b * b

    if norm_a_sq == 0.0 or norm_b_sq == 0.0:
        return 0.0

    denominator = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
    similarity = dot_product / denominator

    # Clamp to [-1.0, 1.0] to protect against floating point rounding imprecision
    return max(-1.0, min(1.0, similarity))
