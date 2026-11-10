
"""Baseline KNN spectral : prédiction de fingerprint à partir des spectres voisins."""

import numpy as np
from scipy import sparse


def find_neighbors(query_spectrum, train_spectra, k=10):
    """Retourne les indices et similarités des k spectres les plus proches."""

    if sparse.issparse(query_spectrum):
        query_spectrum = query_spectrum.toarray()

    query_spectrum = np.asarray(query_spectrum).reshape(-1)

    similarities = train_spectra @ query_spectrum
    similarities = np.asarray(similarities).reshape(-1)

    k = min(k, len(similarities))

    if k <= 0:
        raise ValueError("k doit être strictement positif.")

    indices = np.argpartition(similarities, -k)[-k:]
    indices = indices[np.argsort(similarities[indices])[::-1]]

    return indices, similarities[indices]


def predict_fingerprint(
    query_spectrum,
    train_spectra,
    train_fingerprints,
    k=10
):
    """Prédit un fingerprint par moyenne pondérée des voisins."""

    indices, similarities = find_neighbors(
        query_spectrum, train_spectra, k=k
    )

    weights = np.maximum(similarities, 0)

    if weights.sum() == 0:
        weights = np.ones(len(weights))

    weights = weights / weights.sum()

    fingerprints = train_fingerprints[indices]

    predicted_fp = np.average(
        fingerprints.astype(np.float32),
        axis=0,
        weights=weights
    )

    return predicted_fp
