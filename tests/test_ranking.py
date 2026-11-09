import numpy as np

from casmi.ranking import (
    tanimoto_scores,
    rank_candidates,
    aggregate_fps,
)


def test_tanimoto_identical():
    assert tanimoto_scores(
        np.array([1, 0, 1]),
        np.array([[1, 0, 1]])
    )[0] == 1


def test_tanimoto_disjoint():
    assert tanimoto_scores(
        np.array([1, 0, 0]),
        np.array([[0, 1, 0]])
    )[0] == 0


def test_rank_candidates():
    pred = np.array([1, 0, 0], dtype=np.float32)
    fps = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0]
    ], dtype=np.float32)

    ranked, scores = rank_candidates(pred, [0, 1, 2], fps)

    assert ranked.tolist() == [1, 2, 0]
    assert scores[0] == 1.0


def test_aggregate_mean():
    fps = np.array([[1, 0], [0, 1]])
    result = aggregate_fps(fps)
    np.testing.assert_allclose(result, [0.5, 0.5])


def test_aggregate_weighted():
    fps = np.array([[1, 0], [0, 1]])
    result = aggregate_fps(fps, method="weighted", weights=[3, 1])
    np.testing.assert_allclose(result, [0.75, 0.25])


def test_rank_empty_candidates():
    ranked, scores = rank_candidates(
        np.array([1, 0]), [], np.array([[1, 0]])
    )
    assert len(ranked) == 0
    assert len(scores) == 0
