
"""Construction des requêtes et évaluation du baseline KNN."""

import numpy as np
import pandas as pd

from casmi.io import load_meta, load_spectra
from casmi.split import load_split
from casmi.mass import compute_neutral_masses
from casmi.features import bin_spectra
from casmi.catalog import load_catalog, get_candidates
from casmi.fingerprints import load_catalog_fps
from casmi.ranking import rank_candidates
from casmi.metrics import mrr_at_k, recall_at_k
from casmi.models.knn import predict_fingerprint


def _select_dev_queries(meta, split):
    """Sélectionne les spectres des molécules dev."""
    dev_keys = split.loc[
        (split["split"] == "val") & split["dev"],
        "inchikey14"
    ]

    queries = meta[meta["inchikey14"].isin(dev_keys)].copy()
    queries["query_id"] = queries["inchikey14"]
    queries["true_inchikey14"] = queries["inchikey14"]
    queries["neutral_mass"] = compute_neutral_masses(queries)

    return queries.reset_index(drop=True)


def build_queries(split="dev"):
    """Construit les requêtes de validation, une ligne par spectre."""
    if split != "dev":
        raise ValueError("Pour le moment, seul split='dev' est disponible.")

    meta = load_meta(
        columns=["inchikey14", "precursor_mz", "adduct"]
    )
    split_df = load_split()

    queries = _select_dev_queries(meta, split_df)
    spectra = load_spectra(queries["row_id"].tolist())

    return queries.merge(
        spectra, on="row_id", how="left", validate="one_to_one"
    )


def evaluate(scorer=predict_fingerprint, split="dev",
             ppm=10, train_size=10000, k_neighbors=10,
             k_rank=25, random_state=42):
    """Évalue un modèle sur les 2000 molécules dev.

    scorer(query_spectrum, train_spectra, train_fingerprints, k=...)
    doit renvoyer un fingerprint prédit.
    """
    if split != "dev":
        raise ValueError("Pour le moment, seul split='dev' est disponible.")

    # 1. Préparation des requêtes dev : une par molécule
    queries = build_queries(split)
    queries = queries.drop_duplicates("query_id").reset_index(drop=True)

    X_dev = bin_spectra(
        queries["ms2_mzs"].tolist(),
        queries["ms2_normalized_intensities"].tolist()
    )

    # 2. Préparation d'un échantillon train
    meta = load_meta(columns=["inchikey14"])
    split_df = load_split()

    train_keys = split_df.loc[
        split_df["split"] == "train", "inchikey14"
    ]

    train_meta = meta[meta["inchikey14"].isin(train_keys)]
    train_meta = train_meta.sample(
        n=min(train_size, len(train_meta)),
        random_state=random_state
    ).reset_index(drop=True)

    train_spectra_df = load_spectra(train_meta["row_id"].tolist())
    train_meta = train_meta.merge(
        train_spectra_df, on="row_id", how="left",
        validate="one_to_one"
    )

    X_train = bin_spectra(
        train_meta["ms2_mzs"].tolist(),
        train_meta["ms2_normalized_intensities"].tolist()
    )

    # 3. Fingerprints des molécules train
    catalog = load_catalog()
    key_to_idx = catalog.set_index("inchikey14")["cat_idx"]

    train_cat_idx = train_meta["inchikey14"].map(key_to_idx).to_numpy()
    train_cat_idx = train_cat_idx.astype(int)

    catalog_fp = load_catalog_fps(mmap=True)
    train_fp = catalog_fp[train_cat_idx]

    # 4. Masse du catalogue pour filtrer les candidats
    catalog_masses = catalog["exact_mass"].to_numpy()
    catalog_keys = catalog["inchikey14"].to_numpy()

    # 5. Prédiction et classement
    true_keys = []
    ranked_keys = []

    for i in range(len(queries)):
        query_spectrum = X_dev[i]

        predicted_fp = scorer(
            query_spectrum, X_train, train_fp, k=k_neighbors
        )

        candidates = get_candidates(
            queries.loc[i, "neutral_mass"],
            ppm=ppm,
            masses=catalog_masses
        )

        if len(candidates) == 0:
            ranked_keys.append([])
        else:
            ranked_idx = rank_candidates(
                predicted_fp, candidates, catalog_fp
            )
            ranked_keys.append(catalog_keys[ranked_idx].tolist())

        true_keys.append(queries.loc[i, "true_inchikey14"])

    # 6. Métriques
    return {
        "split": split,
        "n_queries": len(queries),
        "n_train_spectra": len(train_meta),
        "ppm": ppm,
        "k_neighbors": k_neighbors,
        "mrr_at_25": mrr_at_k(true_keys, ranked_keys, k=k_rank),
        "recall_at_1": recall_at_k(true_keys, ranked_keys, k=1),
        "recall_at_5": recall_at_k(true_keys, ranked_keys, k=5),
        "recall_at_25": recall_at_k(true_keys, ranked_keys, k=25),
    }
