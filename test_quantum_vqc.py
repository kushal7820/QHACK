"""Tests for the quantum-only VQC module. Run: pytest -q tests/"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantum_preprocessing import (  # noqa: E402
    PreprocessConfig, QuantumFeaturePreprocessor, encode_target, make_splits,
    make_synthetic_dataset, validate_dataframe,
)
from vqc_model import QuantumPipeline, VQCClassifier, VQCConfig, compute_metrics  # noqa: E402
import train_vqc  # noqa: E402


@pytest.fixture(scope="module")
def data():
    df, _ = validate_dataframe(make_synthetic_dataset(200, seed=1))
    y = encode_target(df["Cancer"], 1)
    sp = make_splits(y, seed=0)
    pre = QuantumFeaturePreprocessor(PreprocessConfig(n_features=2, seed=0))
    X_tr = pre.fit_transform(df.iloc[sp["train"]], y[sp["train"]])
    return df, y, sp, pre, X_tr, pre.transform(df.iloc[sp["val"]]), pre.transform(df.iloc[sp["test"]])


def small_cfg(**kw):
    base = dict(n_qubits=2, n_layers=1, epochs=4, patience=10, batch_size=32, seed=0)
    base.update(kw)
    return VQCConfig(**base)


def test_1_circuit_construction_and_execution():
    m = VQCClassifier(small_cfg(n_qubits=3, n_layers=2, reupload=True))
    assert "RY" in m.draw() and "CNOT" in m.draw() or "╭●" in m.draw()
    m.weights_ = np.random.default_rng(0).normal(size=(2, 3, 3))
    z = m.decision_function(np.random.default_rng(1).uniform(-np.pi, np.pi, (5, 3)))
    assert z.shape == (5,) and np.all(np.abs(z) <= 1 + 1e-9)


def test_2_prediction_shapes_and_values(data):
    _, y, sp, _, X_tr, X_va, _ = data
    m = VQCClassifier(small_cfg()).fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    s, p = m.predict_scores(X_va), m.predict(X_va)
    assert s.shape == p.shape == (len(X_va),)
    assert np.all((s >= 0) & (s <= 1)) and set(np.unique(p)) <= {0, 1}
    with pytest.raises(ValueError):
        m.predict_scores(np.zeros((3, 5)))  # wrong feature count


def test_3_parameters_actually_update(data):
    _, y, sp, _, X_tr, X_va, _ = data
    m = VQCClassifier(small_cfg(epochs=1))
    init = np.random.default_rng(0).normal(0, m.config.init_scale, (1, 2, 3))
    m.fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    assert not np.allclose(m.weights_, init)


def test_4_loss_tracking(data):
    _, y, sp, _, X_tr, X_va, _ = data
    m = VQCClassifier(small_cfg(epochs=5)).fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    h = m.history_
    assert len(h["train_loss"]) == len(h["val_loss"]) == m.training_info_["epochs_completed"] == 5
    assert np.all(np.isfinite(h["train_loss"])) and np.all(np.isfinite(h["val_loss"]))


def test_5_early_stopping_restores_best(data):
    _, y, sp, _, X_tr, X_va, _ = data
    m = VQCClassifier(small_cfg(epochs=40, patience=2, learning_rate=0.5, min_delta=0.0))
    m.fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    info = m.training_info_
    assert info["epochs_completed"] <= 40
    assert info["best_monitored_loss"] == pytest.approx(min(m.history_["val_loss"]))
    # restored parameters reproduce the best validation loss
    best_loss = m._eval_loss(m.weights_, X_va, y[sp["val"]].astype(float),
                             np.where(y[sp["val"]] == 1, *(
                                 [len(sp["train"]) / (2.0 * np.sum(y[sp["train"]] == 1)),
                                  len(sp["train"]) / (2.0 * np.sum(y[sp["train"]] == 0))])))
    assert best_loss == pytest.approx(info["best_monitored_loss"], abs=1e-9)


def test_6_preprocessing_no_leakage_and_order(data):
    df, y, sp, pre, X_tr, _, _ = data
    train_df = df.iloc[sp["train"]]
    scaler = pre.column_transformer_.named_transformers_["num"].named_steps["sc"]
    age = train_df["Age"]
    expected = age.fillna(age.median()).mean()  # train-only median imputation, then mean
    assert scaler.mean_[0] == pytest.approx(expected, rel=1e-9)
    all_age = df["Age"]
    assert scaler.mean_[0] != pytest.approx(all_age.fillna(all_age.median()).mean(), rel=1e-9)  # not fit on all rows
    a = pre.transform(df.iloc[sp["test"]])
    b = pre.transform(df.iloc[sp["test"]][list(reversed(df.columns))])  # column order must not matter
    np.testing.assert_allclose(a, b)
    assert a.min() >= -np.pi - 1e-9 and a.max() <= np.pi + 1e-9
    with pytest.raises(ValueError):
        pre.transform(df.drop(columns=["Benzene"]))
    assert "Cancer" not in pre.input_columns


def test_6b_group_split_has_no_patient_overlap():
    y = np.tile([0, 1], 150)
    groups = np.repeat(np.arange(100), 3)
    sp = make_splits(y, groups, seed=0)
    g = {k: set(groups[v]) for k, v in sp.items()}
    assert not (g["train"] & g["val"]) and not (g["train"] & g["test"]) and not (g["val"] & g["test"])


def test_7_unseen_samples(data):
    df, y, sp, pre, X_tr, X_va, X_te = data
    m = VQCClassifier(small_cfg()).fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    s = m.predict_scores(X_te)
    assert np.all(np.isfinite(s)) and len(s) == len(sp["test"])
    assert compute_metrics(y[sp["test"]], s, m.threshold_)["n"] == len(sp["test"])


def test_8_save_load_roundtrip(data, tmp_path):
    df, y, sp, pre, X_tr, X_va, _ = data
    m = VQCClassifier(small_cfg()).fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    pipe = QuantumPipeline(pre, m)
    pipe.save(tmp_path)
    loaded = QuantumPipeline.load(tmp_path)
    raw = df.iloc[sp["test"]]
    np.testing.assert_allclose(pipe.predict_scores(raw), loaded.predict_scores(raw), atol=1e-10)
    np.testing.assert_array_equal(pipe.predict(raw), loaded.predict(raw))
    assert loaded.model.threshold_ == m.threshold_


def test_9_reproducibility(data):
    _, y, sp, _, X_tr, X_va, _ = data
    a = VQCClassifier(small_cfg(seed=7)).fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    b = VQCClassifier(small_cfg(seed=7)).fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    c = VQCClassifier(small_cfg(seed=8)).fit(X_tr, y[sp["train"]], X_va, y[sp["val"]], verbose=False)
    np.testing.assert_allclose(a.weights_, b.weights_)
    assert not np.allclose(a.weights_, c.weights_)


def test_10_end_to_end(tmp_path):
    res = train_vqc.run([
        "--synthetic", "--n-synthetic", "200", "--n-qubits", "2", "--n-layers", "1", "--epochs", "3",
        "--n-runs", "2", "--robustness", "--noise-levels", "0.05", "--output-dir", str(tmp_path), "--quiet",
    ])
    for f in ("metrics.json", "config.json", "loss_curves.png", "confusion_matrix_test.png",
              "model/vqc_params.npz", "preprocessing/preprocessor.joblib"):
        assert (tmp_path / f).exists(), f
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["n_failed_runs"] == 0 and "SYNTHETIC" in saved["data_source"]
    assert 0 <= res["test"]["balanced_accuracy"] <= 1


def test_real_data_requires_explicit_positive_label(tmp_path):
    csv = tmp_path / "d.csv"
    make_synthetic_dataset(60, 0).to_csv(csv, index=False)
    with pytest.raises(SystemExit):
        train_vqc.run(["--data", str(csv), "--output-dir", str(tmp_path / "o"), "--quiet"])
