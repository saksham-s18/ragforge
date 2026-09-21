import pytest

from ragforge.core.similarity import cosine_similarity
from ragforge.domain.exceptions import VectorDimensionMismatchError


def test_cosine_similarity_identical_vectors() -> None:
    """Verify identical non-zero vectors yield a cosine similarity of 1.0."""
    v1 = [1.0, 2.0, 3.0]
    v2 = [1.0, 2.0, 3.0]
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Verify orthogonal vectors yield a cosine similarity of 0.0."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v2) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors() -> None:
    """Verify diametrically opposite vectors yield a cosine similarity of -1.0."""
    v1 = [2.0, 0.0]
    v2 = [-4.0, 0.0]
    assert cosine_similarity(v1, v2) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_behavior() -> None:
    """Verify zero vectors safely return 0.0 without divide-by-zero errors."""
    zero_vec = [0.0, 0.0, 0.0]
    non_zero_vec = [1.0, 2.0, 3.0]

    assert cosine_similarity(zero_vec, non_zero_vec) == 0.0
    assert cosine_similarity(non_zero_vec, zero_vec) == 0.0
    assert cosine_similarity(zero_vec, zero_vec) == 0.0


def test_cosine_similarity_dimension_mismatch() -> None:
    """Verify mismatched vector dimensions raise VectorDimensionMismatchError."""
    v1 = [1.0, 2.0]
    v2 = [1.0, 2.0, 3.0]

    with pytest.raises(VectorDimensionMismatchError, match="Vector dimension mismatch"):
        cosine_similarity(v1, v2)


def test_cosine_similarity_empty_vectors() -> None:
    """Verify empty vectors safely return 0.0."""
    assert cosine_similarity([], []) == 0.0


def test_cosine_similarity_clamping_boundary() -> None:
    """Verify similarity values are strictly constrained between -1.0 and 1.0."""
    v1 = [1.0000000000000002, 0.0]
    v2 = [1.0000000000000002, 0.0]
    sim = cosine_similarity(v1, v2)
    assert -1.0 <= sim <= 1.0
