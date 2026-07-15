import math

from rag import cosine_similarity


def test_identical_vectors_score_one():
    v = [1.0, 2.0, 3.0]
    assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-9)


def test_orthogonal_vectors_score_zero():
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_opposite_vectors_score_minus_one():
    assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, rel_tol=1e-9)


def test_scale_invariant():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 6.0]
    scaled = [x * 10 for x in b]
    assert math.isclose(cosine_similarity(a, b), cosine_similarity(a, scaled), rel_tol=1e-9)

