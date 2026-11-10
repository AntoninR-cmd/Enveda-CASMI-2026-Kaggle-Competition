"""Comparaison et classement de fingerprints moléculaires."""

import numpy as np


def tanimoto_scores(pred_fp, cand_fps):
    """Score Tanimoto continu entre un fingerprint prédit et les candidats.

    pred_fp : vecteur (n_bits,)
    cand_fps : matrice (n_candidats, n_bits)
    renvoie un vecteur de scores (n_candidats,)
    """
    pred = np.asarray(pred_fp, dtype=np.float32).reshape(-1)
    candidates = np.asarray(cand_fps, dtype=np.float32)

    if candidates.ndim != 2 or candidates.shape[1] != len(pred):
        raise ValueError("Dimensions incompatibles entre pred_fp et cand_fps.")

    intersection = candidates @ pred
    denominator = (
        np.sum(pred ** 2)
        + np.sum(candidates ** 2, axis=1)
        - intersection
    )

    return np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=denominator > 0,
    )


def rank_candidates(pred_fp, candidate_indices, catalog_fp):
    """Classe les cat_idx candidats par score Tanimoto décroissant."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

    if len(candidate_indices) == 0:
        return candidate_indices, np.array([], dtype=np.float32)

    scores = tanimoto_scores(pred_fp, catalog_fp[candidate_indices])
    order = np.argsort(scores)[::-1]

    return candidate_indices[order], scores[order]


def aggregate_fps(fps, method="mean", weights=None):
    """Agrège plusieurs fingerprints en un fingerprint prédit."""
    fps = np.asarray(fps, dtype=np.float32)

    if fps.ndim != 2 or len(fps) == 0:
        raise ValueError("fps doit être une matrice non vide.")

    if method == "mean":
        return fps.mean(axis=0)

    if method == "max":
        return fps.max(axis=0)

    if method == "weighted":
        if weights is None:
            raise ValueError("weights est obligatoire pour method='weighted'.")

        weights = np.asarray(weights, dtype=np.float32)
        if len(weights) != len(fps) or np.any(weights < 0):
            raise ValueError("Poids invalides.")

        if weights.sum() == 0:
            weights = np.ones(len(weights), dtype=np.float32)

        weights = weights / weights.sum()
        return np.average(fps, axis=0, weights=weights)

    raise ValueError("method doit être 'mean', 'max' ou 'weighted'.")
