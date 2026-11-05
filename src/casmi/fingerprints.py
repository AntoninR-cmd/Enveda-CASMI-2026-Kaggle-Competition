"""Fingerprints de Morgan : la cible du modèle et la représentation des candidats.

Usage en ligne de commande, pour (re)construire catalog_fp.npy à partir du catalogue :
    python -m casmi.fingerprints
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from casmi.catalog import N_JOBS, PROCESSED_DIR, load_catalog, map_chunks
from casmi.config import FP_RADIUS, FP_SIZE  # rayon 2 (≈ ECFP4), 2048 bits, binaire

CATALOG_FP_PATH = PROCESSED_DIR / "catalog_fp.npy"

_generator = None


def _morgan_generator():
    # Créé à la demande : le générateur RDKit ne peut pas être envoyé à un autre processus
    global _generator
    if _generator is None:
        _generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
    return _generator


def smiles_to_fp(smiles: str) -> np.ndarray:
    """Fingerprint de Morgan binaire d'un SMILES : tableau uint8 de FP_SIZE valeurs 0 ou 1.

    Lève ValueError si RDKit ne sait pas lire le SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES illisible : {smiles!r}")
    return _morgan_generator().GetFingerprintAsNumPy(mol).astype(np.uint8)


def _packed_fp_chunk(smiles: list[str]) -> list[np.ndarray]:
    RDLogger.DisableLog("rdApp.*")
    # Compressés par 8 (np.packbits) pour limiter les échanges entre processus
    return [np.packbits(smiles_to_fp(s)) for s in smiles]


def compute_fps(smiles, n_jobs: int = N_JOBS) -> np.ndarray:
    """Fingerprints d'une liste de SMILES, calculés en parallèle : matrice (n, FP_SIZE) uint8."""
    packed = map_chunks(_packed_fp_chunk, smiles, n_jobs, chunksize=5_000)
    if not packed:
        return np.zeros((0, FP_SIZE), dtype=np.uint8)
    return np.unpackbits(np.stack(packed), axis=1, count=FP_SIZE)


def build_catalog_fps(
    catalog: pd.DataFrame | None = None, out_path: Path | None = None, n_jobs: int = N_JOBS
) -> np.ndarray:
    """Calcule les fingerprints de tout le catalogue et les enregistre (ligne i = cat_idx i)."""
    catalog = load_catalog() if catalog is None else catalog
    fps = compute_fps(catalog["smiles"], n_jobs)
    out_path = Path(out_path or CATALOG_FP_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, fps)
    return fps


def load_catalog_fps(path: Path | None = None, mmap: bool = False) -> np.ndarray:
    """Fingerprints du catalogue : matrice (n_catalogue, FP_SIZE) uint8, ligne i = cat_idx i.

    Avec mmap=True, le fichier (environ 565 Mo) n'est pas chargé en mémoire : seules
    les lignes utilisées sont lues sur le disque.
    """
    return np.load(Path(path or CATALOG_FP_PATH), mmap_mode="r" if mmap else None)


if __name__ == "__main__":
    t0 = time.perf_counter()
    fps = build_catalog_fps()
    print(f"Fingerprints {fps.shape} écrits dans {CATALOG_FP_PATH} en {time.perf_counter() - t0:.0f} s")
