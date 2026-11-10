"""Génération de la soumission CASMI 2026."""

import numpy as np
import pandas as pd

from casmi.io import load_meta, load_spectra, load_test
from casmi.split import load_split
from casmi.mass import compute_neutral_masses
from casmi.features import bin_spectra
from casmi.catalog import load_catalog, get_candidates
from casmi.fingerprints import load_catalog_fps
from casmi.ranking import rank_candidates, aggregate_fps
from casmi.models.knn import predict_fingerprint


def _format_smiles(smiles_list, max_rank=25):
    """Formate les SMILES en chaîne séparée par des points-virgules."""
    unique_smiles = list(dict.fromkeys(
        str(smiles).strip()
        for smiles in smiles_list
        if pd.notna(smiles) and str(smiles).strip()
    ))

    return ";".join(unique_smiles[:max_rank])


def build_submission(
    train_size=10000,
    k_neighbors=10,
    ppm=10,
    random_state=42,
):
    """Construit une soumission contenant une ligne par molécule test."""

    # Données test
    test = load_test()
    test["neutral_mass"] = compute_neutral_masses(test)

    X_test = bin_spectra(
        test["ms2_mzs"].tolist(),
        test["ms2_normalized_intensities"].tolist()
    )

    # Spectres train
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

    train_spectra = load_spectra(train_meta["row_id"].tolist())
    train_meta = train_meta.merge(
        train_spectra, on="row_id", how="left"
    )

    X_train = bin_spectra(
        train_meta["ms2_mzs"].tolist(),
        train_meta["ms2_normalized_intensities"].tolist()
    )

    # Fingerprints train
    catalog = load_catalog()
    catalog_fp = load_catalog_fps(mmap=True)

    key_to_idx = catalog.set_index("inchikey14")["cat_idx"]
    train_idx = (
        train_meta["inchikey14"]
        .map(key_to_idx)
        .astype(int)
        .to_numpy()
    )
    train_fp = catalog_fp[train_idx]

    # SMILES indexés par cat_idx
    smiles_col = next(
        (col for col in ["smiles", "canonical_smiles", "isomeric_smiles"]
         if col in catalog.columns),
        None
    )

    if smiles_col is None:
        raise ValueError(
            "Aucune colonne SMILES trouvée dans le catalogue : "
            f"{catalog.columns.tolist()}"
        )

    catalog_smiles = catalog.set_index("cat_idx")[smiles_col]
    masses = catalog["exact_mass"].to_numpy()

    # Une prédiction par molécule test
    rows = []

    for molecule_id, group in test.groupby("molecule_id", sort=False):
        spectrum_indices = group.index.to_numpy()

        predicted_fps = [
            predict_fingerprint(
                X_test[idx],
                X_train,
                train_fp,
                k=k_neighbors
            )
            for idx in spectrum_indices
        ]

        predicted_fp = aggregate_fps(
            np.asarray(predicted_fps),
            method="mean"
        )

        neutral_mass = group["neutral_mass"].median()

        candidates = get_candidates(
            neutral_mass,
            ppm=ppm,
            masses=masses
        )

        if len(candidates) == 0:
            raise RuntimeError(
                f"Aucun candidat trouvé pour {molecule_id} "
                f"(masse={neutral_mass:.5f})"
            )

        ranked_idx, _ = rank_candidates(
            predicted_fp,
            candidates,
            catalog_fp
        )

        ranked_smiles = catalog_smiles.loc[ranked_idx].tolist()

        rows.append({
            "molecule_id": molecule_id,
            "smiles": _format_smiles(ranked_smiles, max_rank=25)
        })

    submission = pd.DataFrame(rows, columns=["molecule_id", "smiles"])

    if submission["molecule_id"].duplicated().any():
        raise ValueError("Des molecule_id sont dupliqués.")

    if submission["smiles"].isna().any():
        raise ValueError("Des molécules n'ont aucun SMILES.")

    return submission


def main():
    """Génère submission.csv à la racine du projet."""
    submission = build_submission()
    submission.to_csv("submission.csv", index=False)

    print("Soumission générée : submission.csv")
    print("Dimensions :", submission.shape)
    print("Molécules :", submission["molecule_id"].nunique())
    print("SMILES par molécule :")
    print(submission["smiles"].str.count(";").add(1).describe())


if __name__ == "__main__":
    main()
