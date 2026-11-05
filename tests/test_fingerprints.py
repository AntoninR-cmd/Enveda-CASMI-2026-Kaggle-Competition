import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from casmi.catalog import build_catalog
from casmi.config import FP_RADIUS, FP_SIZE
from casmi.fingerprints import build_catalog_fps, compute_fps, load_catalog_fps, smiles_to_fp

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
THEOBROMINE = "Cn1cnc2c1c(=O)[nH]c(=O)n2C"
ACETIC_ACID = "CC(=O)O"


def test_smiles_to_fp_shape_and_values():
    fp = smiles_to_fp(CAFFEINE)
    assert fp.shape == (FP_SIZE,)
    assert fp.dtype == np.uint8
    assert set(np.unique(fp)) <= {0, 1}
    assert fp.sum() > 0


def test_smiles_to_fp_matches_rdkit_bitvect():
    # Même résultat que le fingerprint RDKit classique, et même Tanimoto
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
    bitvect = generator.GetFingerprint(Chem.MolFromSmiles(CAFFEINE))
    assert np.flatnonzero(smiles_to_fp(CAFFEINE)).tolist() == list(bitvect.GetOnBits())

    a, b = smiles_to_fp(CAFFEINE), smiles_to_fp(THEOBROMINE)
    tanimoto = (a & b).sum() / (a | b).sum()
    expected = DataStructs.TanimotoSimilarity(bitvect, generator.GetFingerprint(Chem.MolFromSmiles(THEOBROMINE)))
    assert tanimoto == pytest.approx(expected)


def test_smiles_to_fp_rejects_invalid():
    with pytest.raises(ValueError):
        smiles_to_fp("pas un smiles")


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_compute_fps(n_jobs):
    fps = compute_fps([CAFFEINE, ACETIC_ACID, THEOBROMINE], n_jobs=n_jobs)
    assert fps.shape == (3, FP_SIZE)
    assert fps.dtype == np.uint8
    np.testing.assert_array_equal(fps[1], smiles_to_fp(ACETIC_ACID))


def test_catalog_fps_rows_follow_cat_idx(tmp_path):
    train = pd.DataFrame({
        "inchikey14": ["CAFE", "ACET", "THEO"],
        "normalized_smiles": [CAFFEINE, ACETIC_ACID, THEOBROMINE],
    })
    train.to_parquet(tmp_path / "train.parquet", index=False)
    catalog = build_catalog(tmp_path / "train.parquet", tmp_path / "catalog.parquet", n_jobs=1)

    build_catalog_fps(catalog, tmp_path / "catalog_fp.npy", n_jobs=1)
    fps = load_catalog_fps(tmp_path / "catalog_fp.npy")

    assert fps.shape == (len(catalog), FP_SIZE)
    for i, smiles in zip(catalog["cat_idx"], catalog["smiles"]):
        np.testing.assert_array_equal(fps[i], smiles_to_fp(smiles))
