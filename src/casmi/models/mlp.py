"""MLP spectre → fingerprint (étape 3 du pipeline).

L'architecture de départ du plan, une boucle d'entraînement avec early stopping, la
prédiction, les métriques et la sauvegarde. Les entrées X sont celles de A
(load_matrices) : [spectre biné | features de métadonnées], en matrice creuse scipy
ou en tableau numpy ; les lignes sont converties en dense par batch seulement.
Les cibles Y sont les fingerprints 0/1 du catalogue.

predict_fp(query_spectra) prédit le fingerprint de spectres bruts (pics et
métadonnées), avec le modèle enregistré dans MODEL_PATH.

Démonstration sur des données aléatoires :
    python -m casmi.models.mlp
"""

import copy
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from casmi.catalog import PROCESSED_DIR
from casmi.config import FP_SIZE

MODEL_PATH = PROCESSED_DIR / "models" / "mlp.pt"  # hors git


class SpectrumToFingerprint(nn.Module):
    """Deux couches cachées (Linear → ReLU → Dropout), puis un logit par bit."""

    def __init__(self, n_inputs: int, hidden: int = 2048, n_bits: int = FP_SIZE, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainConfig:
    hidden: int = 2048
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 1024  # sur CPU, 1 024 est environ 1,5 fois plus rapide par époque que 256
    max_epochs: int = 50
    patience: int = 5  # early stopping : époques sans amélioration de la loss de validation
    pos_weight: float | None = None  # poids des bits à 1 dans la loss (déséquilibre des classes)
    seed: int = 0
    n_threads: int = os.cpu_count() or 1
    train_monitor_size: int = 20_000  # spectres d'entraînement sur lesquels la loss est suivie à chaque époque
    adduct_encoding: str = "onehot"  # encodage de l'adduct des entrées (A) ; sert à predict_fp


def _as_matrix(X):
    """Matrice creuse CSR ou tableau numpy float32 : les deux formes acceptées en entrée."""
    return X.tocsr() if sp.issparse(X) else np.asarray(X, dtype=np.float32)


def _dense(X, rows) -> torch.Tensor:
    """Lignes rows (tableau d'indices ou slice) de X, en tenseur dense float32."""
    part = X[rows]
    if sp.issparse(part):
        part = part.toarray()
    return torch.as_tensor(np.asarray(part, dtype=np.float32))


def _n_rows(X) -> int:
    return X.shape[0]


def _mean_loss(model, loss_fn, X, Y, batch_size) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, _n_rows(X), batch_size):
            rows = slice(start, start + batch_size)
            xb, yb = _dense(X, rows), _dense(Y, rows)
            total += loss_fn(model(xb), yb).item() * len(xb)
    return total / _n_rows(X)


def train_mlp(X_train, Y_train, X_val=None, Y_val=None, config: TrainConfig | None = None, verbose: bool = True):
    """Entraîne un SpectrumToFingerprint ; renvoie (modèle, historique par époque).

    X_train, X_val : matrices creuses scipy ou tableaux numpy (n, n_entrées) ;
    Y_train, Y_val : fingerprints 0/1 (n, n_bits), par exemple en uint8.

    Avec un jeu de validation, l'entraînement s'arrête quand la loss de validation
    ne s'améliore plus pendant config.patience époques, et le modèle renvoyé est
    celui de la meilleure époque. La loss d'entraînement est mesurée en mode
    évaluation (sans dropout) sur config.train_monitor_size spectres tirés une fois
    pour toutes, pour être comparable à la loss de validation sans tout reparcourir.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.n_threads)
    rng = np.random.default_rng(config.seed)

    X_train, Y_train = _as_matrix(X_train), np.asarray(Y_train)
    has_val = X_val is not None
    if has_val:
        X_val, Y_val = _as_matrix(X_val), np.asarray(Y_val)

    n_train = _n_rows(X_train)
    monitor = np.sort(rng.choice(n_train, min(config.train_monitor_size, n_train), replace=False))
    X_monitor, Y_monitor = X_train[monitor], Y_train[monitor]

    n_bits = Y_train.shape[1]
    model = SpectrumToFingerprint(X_train.shape[1], config.hidden, n_bits, config.dropout)
    pos_weight = None if config.pos_weight is None else torch.full((n_bits,), float(config.pos_weight))
    # BCEWithLogitsLoss combine sigmoïde et entropie croisée de façon numériquement stable
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = []
    best_val, best_state, epochs_without_progress = np.inf, None, 0
    for epoch in range(1, config.max_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        order = rng.permutation(n_train)  # mélange à chaque époque
        for start in range(0, n_train, config.batch_size):
            idx = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(_dense(X_train, idx)), _dense(Y_train, idx))
            loss.backward()
            optimizer.step()

        record = {"epoch": epoch, "train_loss": _mean_loss(model, loss_fn, X_monitor, Y_monitor, 4096)}
        if has_val:
            record["val_loss"] = _mean_loss(model, loss_fn, X_val, Y_val, 4096)
            record["val_tanimoto"] = fingerprint_metrics(predict_proba(model, X_val), Y_val)["tanimoto"]
        record["seconds"] = time.perf_counter() - t0
        history.append(record)
        if verbose:
            print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

        if has_val:
            if record["val_loss"] < best_val:
                best_val, best_state, epochs_without_progress = record["val_loss"], copy.deepcopy(model.state_dict()), 0
            else:
                epochs_without_progress += 1
                if epochs_without_progress >= config.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_proba(model: nn.Module, X, batch_size: int = 4096) -> np.ndarray:
    """Probabilité de chaque bit, pour chaque ligne de X (creuse ou dense) : tableau (n, n_bits) float32."""
    X = _as_matrix(X)
    model.eval()
    with torch.no_grad():
        parts = [torch.sigmoid(model(_dense(X, slice(start, start + batch_size))))
                 for start in range(0, _n_rows(X), batch_size)]
    return torch.cat(parts).numpy() if parts else np.zeros((0, model.net[-1].out_features), dtype=np.float32)


def spectra_inputs(spectra, encoder):
    """Entrées du MLP pour des spectres bruts, construites comme load_matrices (A).

    spectra : une ligne par spectre, avec les pics (ms2_mzs, ms2_normalized_intensities)
    et les colonnes de métadonnées META_INPUT_COLUMNS ; encoder : encodeur de A
    (load_encoder). Renvoie une matrice creuse CSR [spectre biné | métadonnées].
    """
    from casmi.features import bin_spectra, meta_features

    X = bin_spectra(spectra["ms2_mzs"], spectra["ms2_normalized_intensities"])
    meta = meta_features(spectra, encoder).to_numpy(dtype=np.float32)
    return sp.hstack([X, sp.csr_matrix(meta)], format="csr")


_default_model = {}


def predict_fp(query_spectra, model: nn.Module | None = None, encoder=None) -> np.ndarray:
    """Fingerprint prédit de chaque spectre : probabilités (n_spectres, n_bits) float32.

    query_spectra : spectres bruts, comme pour spectra_inputs. Sans model, le modèle
    est celui de MODEL_PATH (chargé une seule fois), avec l'encodeur de A
    correspondant à l'encodage de l'adduct de son entraînement. Avec model, il faut
    aussi donner son encodeur.
    """
    if model is None:
        if "model" not in _default_model:
            from casmi.matrices import load_encoder

            _default_model["model"], config = load_model(MODEL_PATH)
            _default_model["encoder"] = load_encoder(config.adduct_encoding)
        model = _default_model["model"]
        encoder = _default_model["encoder"] if encoder is None else encoder
    elif encoder is None:
        raise ValueError("Avec un modèle donné, il faut aussi son encodeur de métadonnées (load_encoder de A).")
    return predict_proba(model, spectra_inputs(query_spectra, encoder))


def fingerprint_metrics(probs: np.ndarray, Y_true: np.ndarray, threshold: float = 0.5) -> dict:
    """Qualité d'un fingerprint prédit, en dehors du MRR.

    - tanimoto : Tanimoto moyen entre la prédiction seuillée et le vrai fingerprint ;
    - precision, recall : précision et rappel des bits à 1, sur l'ensemble des bits.
    """
    pred = np.asarray(probs) >= threshold
    true = np.asarray(Y_true) > 0.5
    common = (pred & true).sum(axis=1)
    union = (pred | true).sum(axis=1)
    tanimoto = np.where(union > 0, common / np.maximum(union, 1), 1.0)
    tp, n_pred, n_true = common.sum(), pred.sum(), true.sum()
    return {
        "tanimoto": float(tanimoto.mean()),
        "precision": float(tp / n_pred) if n_pred else 0.0,
        "recall": float(tp / n_true) if n_true else 0.0,
    }


def save_model(model: SpectrumToFingerprint, config: TrainConfig, path: Path) -> None:
    """Enregistre les poids et les hyperparamètres (hors git, dans data/processed/models/)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first, last = model.net[0], model.net[-1]
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "n_inputs": first.in_features,
        "n_bits": last.out_features,
    }, path)


def load_model(path: Path) -> tuple[SpectrumToFingerprint, TrainConfig]:
    checkpoint = torch.load(Path(path), weights_only=True)
    config = TrainConfig(**checkpoint["config"])
    model = SpectrumToFingerprint(checkpoint["n_inputs"], config.hidden, checkpoint["n_bits"], config.dropout)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


if __name__ == "__main__":
    # Démonstration : entrées aléatoires et fingerprints aléatoires à 2 % de bits à 1
    rng = np.random.default_rng(0)
    X = rng.random((2_000, 500), dtype=np.float32)
    Y = (rng.random((2_000, FP_SIZE)) < 0.02).astype(np.float32)
    model, history = train_mlp(X[:1_600], Y[:1_600], X[1_600:], Y[1_600:],
                               TrainConfig(hidden=512, max_epochs=5, patience=2))
    print(f"{len(history)} époques ; le modèle s'entraîne sans erreur.")
