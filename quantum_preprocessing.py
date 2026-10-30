"""Data validation, splitting and quantum feature preparation for the VOC VQC.

Pipeline (all statistics are fitted on the TRAINING split only):

    raw columns -> impute -> scale numeric / one-hot categorical
                -> reduce to ``n_features`` dims (PCA or SelectKBest)
                -> MinMax-scale to an angle range (default [-pi, pi]) and clip

The final ``n_features`` columns are used as rotation angles, one per qubit.

Research prototype only: nothing here is clinically validated.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

TARGET_COLUMN = "Cancer"
VOC_COLUMNS: List[str] = [
    "Acetone", "Isoprene", "Ethanol", "Formaldehyde", "Benzene", "Toluene", "Methane",
]
NUMERIC_COLUMNS: List[str] = ["Age"] + VOC_COLUMNS
# Meanings of the numeric codes are NOT assumed: these are one-hot encoded as
# nominal categories, so no ordering or semantics is imposed on them.
CATEGORICAL_COLUMNS: List[str] = ["Sex", "Smoking_Status"]
ANGLE_RANGE: Tuple[float, float] = (-float(np.pi), float(np.pi))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_dataframe(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    numeric_columns: Sequence[str] = tuple(NUMERIC_COLUMNS),
    categorical_columns: Sequence[str] = tuple(CATEGORICAL_COLUMNS),
    group_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate and lightly clean the raw dataset.

    Cleaning is deliberately conservative and fully reported:
    * non-numeric entries in numeric columns, +/-inf, negative VOC values and
      ages outside [0, 120] are set to NaN (they are imputed later, using
      training-split statistics only);
    * rows with a missing target (or group id) are dropped;
    * duplicate rows are REPORTED but not removed (identical readings can be
      legitimate; remove them yourself if you know they are data-entry errors).

    Returns the cleaned frame (index reset) and a JSON-serialisable report.
    """
    required = list(numeric_columns) + list(categorical_columns) + [target]
    if group_column:
        required.append(group_column)
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise ValueError(f"Dataset is missing required columns: {absent}. Found: {list(df.columns)}")

    df = df.copy()
    report: Dict[str, Any] = {"n_rows_raw": int(len(df)), "warnings": []}
    report["duplicate_rows"] = int(df.duplicated().sum())

    coerced: Dict[str, int] = {}
    for c in numeric_columns:
        before = int(df[c].isna().sum())
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        coerced[c] = int(df[c].isna().sum()) - before
    report["non_numeric_or_inf_set_to_nan"] = coerced

    invalid: Dict[str, int] = {}
    for c in numeric_columns:
        if c in VOC_COLUMNS:
            bad = df[c] < 0
        elif c == "Age":
            bad = (df[c] < 0) | (df[c] > 120)
        else:
            continue
        invalid[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    report["invalid_values_set_to_nan"] = invalid

    n_before = len(df)
    drop_mask = df[target].isna()
    if group_column:
        drop_mask |= df[group_column].isna()
    df = df.loc[~drop_mask].reset_index(drop=True)
    report["rows_dropped_missing_target_or_group"] = int(n_before - len(df))

    classes = df[target].astype(str).unique()
    if len(classes) != 2:
        raise ValueError(
            f"Target '{target}' must have exactly 2 classes for binary classification; "
            f"found {len(classes)}: {sorted(classes)[:10]}"
        )
    counts = df[target].value_counts()
    report["target_distribution"] = {str(k): int(v) for k, v in counts.items()}
    if counts.min() / counts.sum() < 0.10:
        report["warnings"].append(
            "Minority class is <10% of the data: accuracy is misleading; prefer balanced accuracy / macro-F1."
        )
    if counts.min() < 30:
        report["warnings"].append("Fewer than 30 samples in the minority class: metrics will be very noisy.")

    report["missing_per_column"] = {c: int(df[c].isna().sum()) for c in required}
    report["categorical_values_for_user_review"] = {
        c: sorted(map(str, df[c].dropna().unique()))[:20] for c in categorical_columns
    }
    report["n_rows_clean"] = int(len(df))
    if group_column is None:
        report["warnings"].append(
            "No group/patient column given: rows are assumed independent. If one patient has several "
            "rows, pass --group-column to avoid patient-level leakage."
        )
    return df, report


def encode_target(series: pd.Series, positive_label: Any) -> np.ndarray:
    """Map the raw target to {0, 1}; ``positive_label`` is the raw value meaning class 1.

    The meaning of the codes is never assumed: the caller must state it.
    """
    def norm(v: Any) -> str:
        try:
            f = float(v)
            return str(int(f)) if f.is_integer() else str(f)
        except (TypeError, ValueError):
            return str(v).strip()

    labels = series.map(norm)
    pos = norm(positive_label)
    if pos not in set(labels):
        raise ValueError(f"positive_label={positive_label!r} not found among target values {sorted(set(labels))}")
    return (labels == pos).astype(int).to_numpy()


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_splits(
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Reproducible stratified train/val/test index split (default 70/15/15).

    With ``groups`` (e.g. patient ids) whole groups are kept in one split
    (StratifiedGroupKFold); fractions are then approximate.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    if groups is None:
        trval, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
        train, val = train_test_split(
            trval, test_size=val_size / (1.0 - test_size), stratify=y[trval], random_state=seed
        )
    else:
        groups = np.asarray(groups)
        n1 = max(2, round(1.0 / test_size))
        trval, test = next(StratifiedGroupKFold(n1, shuffle=True, random_state=seed).split(idx, y, groups))
        n2 = max(2, round((1.0 - test_size) / val_size))
        tr_rel, va_rel = next(
            StratifiedGroupKFold(n2, shuffle=True, random_state=seed).split(trval, y[trval], groups[trval])
        )
        train, val = trval[tr_rel], trval[va_rel]
        g = {k: set(groups[v]) for k, v in {"train": train, "val": val, "test": test}.items()}
        assert not (g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"]), "group leakage"
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


# --------------------------------------------------------------------------- #
# Quantum feature preparation
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessConfig:
    """Configuration of the feature-preparation pipeline."""

    n_features: int = 4                      # = number of qubits (one angle per qubit)
    reduction: str = "pca"                   # "pca" | "select_k_best" | "none"
    numeric_columns: List[str] = field(default_factory=lambda: list(NUMERIC_COLUMNS))
    categorical_columns: List[str] = field(default_factory=lambda: list(CATEGORICAL_COLUMNS))
    angle_range: Tuple[float, float] = ANGLE_RANGE
    seed: int = 42


class QuantumFeaturePreprocessor:
    """Fit-on-train, apply-everywhere feature preparation producing qubit angles."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()
        if self.config.reduction not in {"pca", "select_k_best", "none"}:
            raise ValueError("reduction must be 'pca', 'select_k_best' or 'none'")
        self.input_columns: List[str] = list(self.config.numeric_columns) + list(self.config.categorical_columns)
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.reducer_: Any = None
        self.angle_scaler_: Optional[MinMaxScaler] = None
        self.encoded_feature_names_: List[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> "QuantumFeaturePreprocessor":
        """Fit all transformations on TRAINING data only."""
        cfg = self.config
        X = self._select(df)
        num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        cat = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        self.column_transformer_ = ColumnTransformer(
            [("num", num, cfg.numeric_columns), ("cat", cat, cfg.categorical_columns)],
            remainder="drop", sparse_threshold=0.0,
        )
        Z = self.column_transformer_.fit_transform(X)
        self.encoded_feature_names_ = [str(n) for n in self.column_transformer_.get_feature_names_out()]

        if cfg.reduction == "pca":
            if cfg.n_features > Z.shape[1]:
                raise ValueError(f"n_features={cfg.n_features} exceeds encoded dimension {Z.shape[1]}")
            self.reducer_ = PCA(n_components=cfg.n_features, random_state=cfg.seed)
            Z = self.reducer_.fit_transform(Z)
        elif cfg.reduction == "select_k_best":
            if y is None:
                raise ValueError("select_k_best needs the training labels y (training split only)")
            if cfg.n_features > Z.shape[1]:
                raise ValueError(f"n_features={cfg.n_features} exceeds encoded dimension {Z.shape[1]}")
            self.reducer_ = SelectKBest(f_classif, k=cfg.n_features)
            Z = self.reducer_.fit_transform(Z, y)
        else:
            self.reducer_ = None
            if Z.shape[1] != cfg.n_features:
                raise ValueError(
                    f"reduction='none' requires n_features == encoded dimension ({Z.shape[1]}), "
                    f"got n_features={cfg.n_features}. Use 'pca' or 'select_k_best' instead."
                )
        self.angle_scaler_ = MinMaxScaler(feature_range=tuple(cfg.angle_range), clip=True).fit(Z)
        return self

    # ------------------------------------------------------------ transform
    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.input_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input is missing columns required by the preprocessor: {missing}")
        if self.config.numeric_columns and TARGET_COLUMN in self.input_columns:
            raise ValueError("The target must never be an input feature")
        return df[self.input_columns]  # enforces the training-time column order

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the fitted transformations; output is clipped to ``angle_range``."""
        if self.column_transformer_ is None or self.angle_scaler_ is None:
            raise RuntimeError("Preprocessor is not fitted")
        Z = self.column_transformer_.transform(self._select(df))
        if self.reducer_ is not None:
            Z = self.reducer_.transform(Z)
        return np.asarray(self.angle_scaler_.transform(Z), dtype=float)

    def fit_transform(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(df, y).transform(df)

    # ------------------------------------------------------------- reporting
    def describe(self) -> Dict[str, Any]:
        """Relation between original features, transformed dimensions and qubits."""
        info: Dict[str, Any] = {
            "original_features": self.input_columns,
            "encoded_features_after_imputation_scaling_onehot": self.encoded_feature_names_,
            "reduction": self.config.reduction,
            "n_output_features_equals_n_qubits": self.config.n_features,
            "angle_range": list(self.config.angle_range),
        }
        if self.config.reduction == "pca" and self.reducer_ is not None:
            info["pca_explained_variance_ratio"] = [float(v) for v in self.reducer_.explained_variance_ratio_]
            info["pca_total_explained_variance"] = float(self.reducer_.explained_variance_ratio_.sum())
            info["pca_loadings"] = {
                f"component_{i}": {n: float(w) for n, w in zip(self.encoded_feature_names_, row)}
                for i, row in enumerate(self.reducer_.components_)
            }
        if self.config.reduction == "select_k_best" and self.reducer_ is not None:
            mask = self.reducer_.get_support()
            info["selected_encoded_features"] = [n for n, m in zip(self.encoded_feature_names_, mask) if m]
        return info

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"column_transformer": self.column_transformer_, "reducer": self.reducer_,
             "angle_scaler": self.angle_scaler_}, p / "preprocessor.joblib",
        )
        meta = {"config": asdict(self.config), "input_columns": self.input_columns,
                "encoded_feature_names": self.encoded_feature_names_, "description": self.describe()}
        (p / "preprocessor_config.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "QuantumFeaturePreprocessor":
        p = Path(path)
        meta = json.loads((p / "preprocessor_config.json").read_text())
        cfg = dict(meta["config"])
        cfg["angle_range"] = tuple(cfg["angle_range"])
        obj = cls(PreprocessConfig(**cfg))
        parts = joblib.load(p / "preprocessor.joblib")
        obj.column_transformer_ = parts["column_transformer"]
        obj.reducer_ = parts["reducer"]
        obj.angle_scaler_ = parts["angle_scaler"]
        obj.input_columns = meta["input_columns"]
        obj.encoded_feature_names_ = meta["encoded_feature_names"]
        return obj


# --------------------------------------------------------------------------- #
# Synthetic data (software testing ONLY)
# --------------------------------------------------------------------------- #
def make_synthetic_dataset(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Generate SYNTHETIC records with the expected schema for software tests.

    These are NOT patient data and carry no biological meaning. The label is a
    noisy logistic function of a few columns, with ~2% missing feature values.
    Convention used only here: ``Cancer == 1`` is the positive class.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": np.clip(rng.normal(55, 12, n), 18, 90).round(),
        "Sex": rng.integers(0, 2, n),
        "Smoking_Status": rng.integers(0, 3, n),
        "Acetone": rng.lognormal(0.0, 0.4, n),
        "Isoprene": rng.lognormal(0.2, 0.4, n),
        "Ethanol": rng.lognormal(-0.2, 0.5, n),
        "Formaldehyde": rng.lognormal(-0.5, 0.4, n),
        "Benzene": rng.lognormal(-1.0, 0.5, n),
        "Toluene": rng.lognormal(-0.8, 0.5, n),
        "Methane": rng.lognormal(0.5, 0.3, n),
    })
    logit = (1.2 * (df["Acetone"] - 1.0) + 1.0 * (df["Benzene"] - 0.4) + 0.8 * (df["Toluene"] - 0.5)
             + 0.03 * (df["Age"] - 55) + 0.3 * (df["Smoking_Status"] == 2) - 0.2)
    df["Cancer"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-2.0 * logit))).astype(int)
    for c in NUMERIC_COLUMNS:
        df.loc[rng.random(n) < 0.02, c] = np.nan
    return df
