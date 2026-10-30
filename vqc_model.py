"""Variational Quantum Classifier (VQC) built with PennyLane.

Circuit (n qubits, L layers, one input angle per qubit):

    for layer l in 0..L-1:
        [if l == 0 or reupload]  RY(x_i) on every qubit i      (angle encoding)
        Rot(phi, theta, omega) on every qubit i                (trainable, 3 params/qubit)
        CNOT entangling pattern ("ring", "linear" or "none")
    output: z = <Z_0> in [-1, 1]

Score mapping: p = (1 + z) / 2 in [0, 1]. This is a bounded model score, NOT a
calibrated clinical probability. Training loss: (class-weighted) binary
cross-entropy on p, clipped to [eps, 1-eps] for numerical stability.

The simulator (``default.qubit``) cost grows as 2^n_qubits in memory/time, so
this is a classical simulation; no quantum speed-up or advantage is claimed.

Research prototype only: not a validated diagnostic system.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from quantum_preprocessing import QuantumFeaturePreprocessor

_OPTIMIZERS = {
    "adam": qml.AdamOptimizer,
    "sgd": qml.GradientDescentOptimizer,
    "adagrad": qml.AdagradOptimizer,
    "rmsprop": qml.RMSPropOptimizer,
    "nesterov": qml.NesterovMomentumOptimizer,
}


def environment_info() -> Dict[str, str]:
    """Package versions for reproducibility records."""
    out = {"python": platform.python_version()}
    for pkg in ("pennylane", "numpy", "pandas", "scikit-learn", "matplotlib", "scipy", "joblib"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


@dataclass
class VQCConfig:
    """All VQC hyper-parameters (JSON-serialisable)."""

    n_qubits: int = 4
    n_layers: int = 2
    reupload: bool = False            # re-apply the RY encoding in every layer
    entanglement: str = "ring"        # "ring" | "linear" | "none"
    optimizer: str = "adam"           # adam | sgd | adagrad | rmsprop | nesterov
    learning_rate: float = 0.05
    epochs: int = 40
    batch_size: int = 32
    patience: int = 8                 # early stopping on validation loss
    min_delta: float = 1e-4           # minimum improvement that resets patience
    seed: int = 42
    init_scale: float = 0.5           # std of the normal weight initialisation
    class_weight: Optional[str] = "balanced"   # "balanced" | None
    eps: float = 1e-7                 # probability clipping in the loss

    @property
    def n_parameters(self) -> int:
        return self.n_layers * self.n_qubits * 3


class VQCClassifier:
    """Trainable VQC with a scikit-learn-like API (inputs are angle features)."""

    def __init__(self, config: Optional[VQCConfig] = None) -> None:
        self.config = config or VQCConfig()
        c = self.config
        if not 1 <= c.n_qubits <= 12:
            raise ValueError("n_qubits must be in [1, 12] (simulation cost grows as 2^n)")
        if c.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if c.optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(_OPTIMIZERS)}")
        if c.entanglement not in {"ring", "linear", "none"}:
            raise ValueError("entanglement must be 'ring', 'linear' or 'none'")
        if c.class_weight not in {"balanced", None}:
            raise ValueError("class_weight must be 'balanced' or None")
        self.weights_: Optional[np.ndarray] = None
        self.threshold_: float = 0.5
        self.history_: Dict[str, List[float]] = {}
        self.training_info_: Dict[str, Any] = {}
        self._build_circuit()

    # ------------------------------------------------------------- circuit
    def _pairs(self) -> List[tuple]:
        n, mode = self.config.n_qubits, self.config.entanglement
        if n < 2 or mode == "none":
            return []
        pairs = [(i, i + 1) for i in range(n - 1)]
        if mode == "ring" and n > 2:
            pairs.append((n - 1, 0))
        return pairs

    def _build_circuit(self) -> None:
        c = self.config
        pairs = self._pairs()
        self.device_ = qml.device("default.qubit", wires=c.n_qubits)

        @qml.qnode(self.device_, interface="autograd", diff_method="backprop")
        def circuit(weights, x):
            # x has shape (batch, n_qubits): PennyLane parameter broadcasting
            # evaluates the whole batch in a single simulator call.
            for layer in range(c.n_layers):
                if layer == 0 or c.reupload:
                    for i in range(c.n_qubits):
                        qml.RY(x[:, i], wires=i)
                for i in range(c.n_qubits):
                    qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)
                for a, b in pairs:
                    qml.CNOT(wires=[a, b])
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def draw(self) -> str:
        """ASCII drawing of the circuit for one dummy sample."""
        w = np.zeros((self.config.n_layers, self.config.n_qubits, 3))
        x = np.zeros((1, self.config.n_qubits))
        return qml.draw(self._circuit)(w, x)

    # ---------------------------------------------------------- inference
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.config.n_qubits:
            raise ValueError(f"X must have shape (n_samples, {self.config.n_qubits}); got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or inf")
        return X

    def _expectation(self, weights: Any, X: np.ndarray, chunk: int = 4096) -> Any:
        outs = [self._circuit(weights, X[i:i + chunk]) for i in range(0, len(X), chunk)]
        return outs[0] if len(outs) == 1 else pnp.concatenate(outs)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Expectation value z = <Z_0> in [-1, 1] for every row of X."""
        if self.weights_ is None:
            raise RuntimeError("Model is not trained")
        X = self._check_X(X)
        w = pnp.array(self.weights_, requires_grad=False)
        return np.asarray(self._expectation(w, X), dtype=float).reshape(-1)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Model score p = (1 + z) / 2 in [0, 1] (not a calibrated probability)."""
        return (1.0 + self.decision_function(X)) / 2.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction: 1 if score >= ``threshold_`` else 0."""
        return (self.predict_scores(X) >= self.threshold_).astype(int)

    # ------------------------------------------------------------ training
    def _loss(self, weights: Any, X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> Any:
        p = (1.0 + self._expectation(weights, X)) / 2.0
        p = pnp.clip(p, self.config.eps, 1.0 - self.config.eps)
        ll = y * pnp.log(p) + (1.0 - y) * pnp.log(1.0 - p)
        return -pnp.sum(sw * ll) / pnp.sum(sw)

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        if self.config.class_weight != "balanced":
            return np.ones(len(y))
        n, n1 = len(y), float(np.sum(y))
        n0 = n - n1
        if n0 == 0 or n1 == 0:
            raise ValueError("Training labels contain a single class")
        return np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))

    def _eval_loss(self, weights: Any, X: np.ndarray, y: np.ndarray, sw: np.ndarray) -> float:
        return float(self._loss(pnp.array(weights, requires_grad=False), X, y, sw))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "VQCClassifier":
        """Train on the training labels; validation data drives early stopping.

        Early stopping monitors validation loss (training loss if no validation
        data is given) and restores the best parameters. The decision threshold
        is then chosen on validation data only (balanced accuracy).
        """
        c = self.config
        X_train, y_train = self._check_X(X_train), np.asarray(y_train).astype(float).reshape(-1)
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train have different lengths")
        if not set(np.unique(y_train)) <= {0.0, 1.0}:
            raise ValueError("y_train must be binary {0, 1}")
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val, y_val = self._check_X(X_val), np.asarray(y_val).astype(float).reshape(-1)

        sw_train = self._sample_weights(y_train)
        # Validation loss uses the SAME per-class weights as training (derived from
        # training labels only), so train/val losses are directly comparable.
        w1 = float(sw_train[y_train == 1][0]) if np.any(y_train == 1) else 1.0
        w0 = float(sw_train[y_train == 0][0]) if np.any(y_train == 0) else 1.0
        sw_val = np.where(y_val == 1, w1, w0) if has_val else None
        rng = np.random.default_rng(c.seed)
        pnp.random.seed(c.seed)
        weights = pnp.array(
            rng.normal(0.0, c.init_scale, size=(c.n_layers, c.n_qubits, 3)), requires_grad=True
        )
        opt = _OPTIMIZERS[c.optimizer](stepsize=c.learning_rate)

        hist: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_w, best_loss, best_epoch, bad = np.array(weights), np.inf, -1, 0
        status = "max_epochs_reached"
        t0 = time.perf_counter()
        for epoch in range(c.epochs):
            order = rng.permutation(len(X_train))
            for s in range(0, len(order), c.batch_size):
                b = order[s:s + c.batch_size]
                xb, yb, wb = X_train[b], y_train[b], sw_train[b]
                weights = opt.step(lambda w: self._loss(w, xb, yb, wb), weights)
            tl = self._eval_loss(weights, X_train, y_train, sw_train)
            ta = float(np.mean((self.decision_function_with(weights, X_train) >= 0.0) == (y_train == 1)))
            hist["train_loss"].append(tl)
            hist["train_acc"].append(ta)
            if has_val:
                vl = self._eval_loss(weights, X_val, y_val, sw_val)
                va = float(np.mean((self.decision_function_with(weights, X_val) >= 0.0) == (y_val == 1)))
            else:
                vl, va = float("nan"), float("nan")
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)
            monitored = vl if has_val else tl
            if not np.isfinite(monitored):
                status = "failed_non_finite_loss"
                break
            if monitored < best_loss - c.min_delta:
                best_loss, best_epoch, best_w, bad = monitored, epoch, np.array(weights), 0
            else:
                bad += 1
            if verbose:
                print(f"  epoch {epoch + 1:3d}/{c.epochs}  train_loss={tl:.4f}  val_loss={vl:.4f}  "
                      f"train_acc@z=0={ta:.3f}  val_acc@z=0={va:.3f}")
            if bad >= c.patience:
                status = "early_stopped"
                break
        train_seconds = time.perf_counter() - t0
        if status == "failed_non_finite_loss" and best_epoch < 0:
            raise FloatingPointError("Training diverged: non-finite loss before any valid epoch")

        self.weights_ = np.array(best_w)          # restore best parameters
        self.history_ = hist
        self.threshold_ = 0.5
        if has_val:
            self.threshold_ = self._select_threshold(y_val, self.predict_scores(X_val))
        t1 = time.perf_counter()
        _ = self.predict_scores(X_train[: min(len(X_train), 256)])
        inf_per_sample = (time.perf_counter() - t1) / min(len(X_train), 256)
        self.training_info_ = {
            "n_qubits": c.n_qubits, "n_layers": c.n_layers, "n_parameters": c.n_parameters,
            "feature_dimension": int(X_train.shape[1]),
            "train_seconds": float(train_seconds),
            "inference_seconds_per_sample": float(inf_per_sample),
            "epochs_completed": len(hist["train_loss"]),
            "best_epoch": int(best_epoch) + 1,
            "best_monitored_loss": float(best_loss),
            "monitored": "val_loss" if has_val else "train_loss",
            "convergence_status": status,
            "threshold": float(self.threshold_),
        }
        return self

    def decision_function_with(self, weights: Any, X: np.ndarray) -> np.ndarray:
        """z for explicit weights (used for per-epoch diagnostics)."""
        w = pnp.array(weights, requires_grad=False)
        return np.asarray(self._expectation(w, X), dtype=float).reshape(-1)

    @staticmethod
    def _select_threshold(y_val: np.ndarray, scores: np.ndarray) -> float:
        """Threshold maximising validation balanced accuracy (ties -> closest to 0.5)."""
        grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
        best_t, best_s = 0.5, -1.0
        for t in grid:
            s = balanced_accuracy_score(y_val.astype(int), (scores >= t).astype(int))
            if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
                best_t, best_s = float(t), s
        return best_t

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Save parameters + configuration (the circuit is rebuilt on load)."""
        if self.weights_ is None:
            raise RuntimeError("Model is not trained")
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "vqc_params.npz", weights=self.weights_)
        meta = {"config": asdict(self.config), "threshold": self.threshold_, "history": self.history_,
                "training_info": self.training_info_, "environment": environment_info()}
        (p / "vqc_config.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VQCClassifier":
        p = Path(path)
        meta = json.loads((p / "vqc_config.json").read_text())
        obj = cls(VQCConfig(**meta["config"]))
        obj.weights_ = np.load(p / "vqc_params.npz")["weights"]
        obj.threshold_ = float(meta["threshold"])
        obj.history_ = meta["history"]
        obj.training_info_ = meta["training_info"]
        return obj


# --------------------------------------------------------------------------- #
# End-to-end pipeline (raw DataFrame -> prediction), handy for a dashboard
# --------------------------------------------------------------------------- #
class QuantumPipeline:
    """Fitted preprocessor + trained VQC; takes raw DataFrames."""

    def __init__(self, preprocessor: QuantumFeaturePreprocessor, model: VQCClassifier) -> None:
        if preprocessor.config.n_features != model.config.n_qubits:
            raise ValueError("preprocessor.n_features must equal model.n_qubits")
        self.preprocessor, self.model = preprocessor, model

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_scores(self.preprocessor.transform(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.preprocessor.transform(df))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        self.preprocessor.save(p / "preprocessing")
        self.model.save(p / "model")

    @classmethod
    def load(cls, path: str | Path) -> "QuantumPipeline":
        p = Path(path)
        return cls(QuantumFeaturePreprocessor.load(p / "preprocessing"), VQCClassifier.load(p / "model"))


# --------------------------------------------------------------------------- #
# Evaluation utilities
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Classification metrics at ``threshold``; ROC-AUC only if both classes exist."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else None,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],  # rows=true 0/1, cols=pred 0/1
    }


def generalization_gap(train_metrics: Dict[str, Any], other_metrics: Dict[str, Any],
                       keys: Sequence[str] = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")) -> Dict[str, Any]:
    """gap = metric(train) - metric(other). Positive => better on train than on other.

    A diagnostic indicator only: a small gap does not prove the model is good
    (it may underfit) and a large gap does not by itself prove memorisation.
    """
    out: Dict[str, Any] = {}
    for k in keys:
        a, b = train_metrics.get(k), other_metrics.get(k)
        out[k] = None if a is None or b is None else float(a - b)
    return out


def perturbation_sensitivity(
    pipeline: QuantumPipeline,
    df: pd.DataFrame,
    y: np.ndarray,
    columns: Sequence[str],
    relative_sigmas: Sequence[float],
    n_repeats: int = 5,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """SIMULATED stress test: multiplicative Gaussian noise on chosen input columns.

    ``relative_sigmas`` are hypothetical stress levels chosen by the user, NOT
    measured sensor tolerances. Labels are never perturbed. Pass validation
    data only (never the test set) if results influence any decision.
    """
    rng = np.random.default_rng(seed)
    base_scores = pipeline.predict_scores(df)
    thr = pipeline.model.threshold_
    base_pred = (base_scores >= thr).astype(int)
    rows = []
    for sigma in relative_sigmas:
        bal, flips, auc = [], [], []
        for _ in range(n_repeats):
            noisy = df.copy()
            for c in columns:
                noisy[c] = np.clip(noisy[c] * (1.0 + sigma * rng.standard_normal(len(noisy))), 0, None)
            s = pipeline.predict_scores(noisy)
            m = compute_metrics(y, s, thr)
            bal.append(m["balanced_accuracy"])
            flips.append(float(np.mean((s >= thr).astype(int) != base_pred)))
            if m["roc_auc"] is not None:
                auc.append(m["roc_auc"])
        rows.append({
            "relative_sigma": float(sigma),
            "balanced_accuracy_mean": float(np.mean(bal)), "balanced_accuracy_std": float(np.std(bal)),
            "prediction_flip_rate_mean": float(np.mean(flips)),
            "roc_auc_mean": float(np.mean(auc)) if auc else None,
            "note": "simulated perturbation, not real sensor validation",
        })
    return rows
