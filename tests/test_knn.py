import numpy as np
from scipy import sparse

from casmi.models.knn import find_neighbors, predict_fingerprint


def test_find_neighbors():
    train = sparse.csr_matrix([
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
    ])
    query = np.array([1, 0, 0])

    indices, similarities = find_neighbors(query, train, k=2)

    assert 0 in indices
    assert 2 in indices
    assert similarities[0] == 1.0
    assert similarities[1] == 1.0
    assert len(indices) == 2


def test_predict_fingerprint():
    train = sparse.csr_matrix([
        [1, 0, 0],
        [0, 1, 0],
    ])
    fingerprints = np.array([
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

    query = np.array([1, 0, 0])
    pred = predict_fingerprint(
        query, train, fingerprints, k=2
    )

    assert pred.shape == (3,)
    assert np.all(pred >= 0)
    assert np.all(pred <= 1)


def test_find_neighbors_k_larger_than_train():
    train = sparse.csr_matrix([[1, 0], [0, 1]])
    indices, similarities = find_neighbors(
        np.array([1, 0]), train, k=10
    )

    assert len(indices) == 2
