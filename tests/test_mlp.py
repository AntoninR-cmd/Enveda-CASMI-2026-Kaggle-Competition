import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from casmi.config import FP_SIZE  # noqa: E402
from casmi.features import fit_meta_encoder, n_bins  # noqa: E402
from casmi.models.mlp import (  # noqa: E402
    SpectrumToFingerprint,
    TrainConfig,
    fingerprint_metrics,
    load_model,
    predict_fp,
    predict_proba,
    save_model,
    spectra_inputs,
    train_mlp,
)


def random_data(n=64, n_inputs=100, n_bits=256, density=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_inputs), dtype=np.float32)
    Y = (rng.random((n, n_bits)) < density).astype(np.float32)
    return X, Y


def test_forward_shape():
    model = SpectrumToFingerprint(n_inputs=100, hidden=32, n_bits=2048)
    assert model(torch.zeros(5, 100)).shape == (5, 2048)


def test_can_overfit_a_tiny_dataset():
    # Contrôle du plan : si le modèle ne sur-apprend pas un tout petit échantillon, il y a un bug
    X, Y = random_data()
    config = TrainConfig(hidden=256, dropout=0.0, lr=1e-3, batch_size=16, max_epochs=150, n_threads=2)
    model, history = train_mlp(X, Y, config=config, verbose=False)

    assert history[-1]["train_loss"] < history[0]["train_loss"] / 10
    assert fingerprint_metrics(predict_proba(model, X), Y)["tanimoto"] > 0.95


def test_early_stopping_keeps_best_epoch():
    # Sur des données aléatoires, un modèle qui apprend vite finit par apprendre le train
    # par cœur : la loss de validation remonte, et l'arrêt anticipé doit se déclencher
    X, Y = random_data(n=200)
    config = TrainConfig(hidden=256, dropout=0.0, lr=1e-2, batch_size=16, max_epochs=100, patience=3, n_threads=2)
    model, history = train_mlp(X[:150], Y[:150], X[150:], Y[150:], config=config, verbose=False)

    assert len(history) < 100
    best = min(h["val_loss"] for h in history)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        final = loss_fn(model(torch.as_tensor(X[150:])), torch.as_tensor(Y[150:])).item()
    assert final == pytest.approx(best, rel=1e-4)


def test_predict_proba_range():
    X, Y = random_data(n=10)
    model, _ = train_mlp(X, Y, config=TrainConfig(hidden=16, max_epochs=1, n_threads=1), verbose=False)
    probs = predict_proba(model, X)
    assert probs.shape == Y.shape
    assert probs.min() >= 0 and probs.max() <= 1


def test_fingerprint_metrics_hand_computed():
    probs = np.array([[0.9, 0.8, 0.1, 0.0], [0.6, 0.0, 0.0, 0.7]])
    truth = np.array([[1, 0, 1, 0], [1, 0, 0, 1]])
    m = fingerprint_metrics(probs, truth)
    # Ligne 1 : prédit {0, 1}, vrai {0, 2} → 1 commun / 3 → 1/3 ; ligne 2 : identique → 1
    assert m["tanimoto"] == pytest.approx((1 / 3 + 1) / 2)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)


def test_save_and_load(tmp_path):
    X, Y = random_data(n=20)
    config = TrainConfig(hidden=16, max_epochs=2, n_threads=1)
    model, _ = train_mlp(X, Y, config=config, verbose=False)

    save_model(model, config, tmp_path / "models" / "mlp.pt")
    loaded, loaded_config = load_model(tmp_path / "models" / "mlp.pt")

    assert loaded_config == config
    np.testing.assert_allclose(predict_proba(loaded, X), predict_proba(model, X), rtol=1e-6)


def test_sparse_inputs_give_the_same_model_as_dense():
    # Les matrices de A sont creuses : l'entraînement doit donner exactement le même modèle
    X, Y = random_data(n=80)
    X[X < 0.8] = 0  # 80 % de zéros, comme un spectre biné
    config = TrainConfig(hidden=32, max_epochs=3, batch_size=16, n_threads=1)
    dense_model, dense_history = train_mlp(X, Y, X[:20], Y[:20], config=config, verbose=False)
    sparse_model, sparse_history = train_mlp(sp.csr_matrix(X), Y.astype(np.uint8), sp.csr_matrix(X[:20]), Y[:20],
                                             config=config, verbose=False)

    assert [h["val_loss"] for h in sparse_history] == pytest.approx([h["val_loss"] for h in dense_history])
    np.testing.assert_allclose(predict_proba(sparse_model, sp.csr_matrix(X)), predict_proba(dense_model, X), rtol=1e-5)


def make_spectra(n=40, seed=0):
    """Spectres bruts factices : pics et métadonnées, comme les lignes du train."""
    rng = np.random.default_rng(seed)
    positive = np.arange(n) % 3 != 0
    ce = rng.choice([20.0, 40.0, 60.0], n)
    n_peaks = rng.integers(3, 30, n)
    return pd.DataFrame({
        "adduct": np.where(positive, "[M+H]+", "[M-H]-"),
        "ionization_mode": np.where(positive, "positive", "negative"),
        "instrument_type": "timsTOF Pro",
        "precursor_mz": rng.uniform(150, 600, n),
        "collision_energy_ev": [[v] for v in ce],
        "collision_energy_orig": [f"{v:g}" for v in ce],
        "collision_energy_orig_units": "eV",
        "ms2_mzs": [np.sort(rng.uniform(50, 600, k)) for k in n_peaks],
        "ms2_normalized_intensities": [rng.uniform(0.01, 1, k) for k in n_peaks],
    })


def test_predict_fp_on_raw_spectra():
    spectra = make_spectra()
    encoder = fit_meta_encoder(spectra, min_adduct_count=10)
    X = spectra_inputs(spectra, encoder)
    assert sp.issparse(X)
    assert X.shape[0] == len(spectra) and X.shape[1] > n_bins()  # spectre biné, puis métadonnées

    model = SpectrumToFingerprint(n_inputs=X.shape[1], hidden=16)
    probs = predict_fp(spectra, model=model, encoder=encoder)
    assert probs.shape == (len(spectra), FP_SIZE)
    assert probs.min() >= 0 and probs.max() <= 1
    np.testing.assert_allclose(probs, predict_proba(model, X))


def test_predict_fp_needs_the_encoder_of_a_given_model():
    model = SpectrumToFingerprint(n_inputs=10, hidden=4)
    with pytest.raises(ValueError):
        predict_fp(make_spectra(n=3), model=model)
