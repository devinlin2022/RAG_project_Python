"""Vector index, RRF fusion and the chunk library, without any network call."""

import numpy as np
import pytest

from rag.index.dense import FlatIndex, normalize
from rag.index.fusion import rrf


def test_normalize_makes_unit_rows():
    matrix = normalize(np.array([[3.0, 4.0]], dtype="float32"))
    assert pytest.approx(float(np.linalg.norm(matrix[0])), abs=1e-6) == 1.0


def test_dense_returns_similar_first():
    index = FlatIndex(np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype="float32"))
    hits = index.search([1, 0], 2)
    assert [row for row, _ in hits] == [0, 1]
    assert hits[0][1] > hits[1][1]


def test_dense_mask_excludes_rows():
    index = FlatIndex(np.array([[1, 0], [1, 0.01]], dtype="float32"))
    hits = index.search([1, 0], 2, allowed=np.array([False, True]))
    assert [row for row, _ in hits] == [1]


def test_dense_dimension_mismatch_is_explicit():
    index = FlatIndex(np.array([[1, 0, 0]], dtype="float32"))
    with pytest.raises(ValueError):
        index.search([1, 0], 1)


def test_rrf_rewards_agreement():
    dense = [7, 1, 2]
    sparse = [2, 7, 9]
    fused = rrf([dense, sparse], k=60)
    # 7 is ranked 1st and 2nd; 2 is 3rd and 1st, so 7 wins by one rank slot
    assert fused[7] > fused[2]
    assert fused[1] > fused[9]


def test_rrf_handles_empty_rail():
    fused = rrf([[1, 2], []], k=60)
    assert set(fused) == {1, 2}
