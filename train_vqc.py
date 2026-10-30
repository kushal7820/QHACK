"""Train and evaluate the quantum-only VQC on breath VOC data.

Examples
--------
    python train_vqc.py --data breath_voc.csv --positive-label 1
    python train_vqc.py --synthetic --n-runs 3          # software test only, no real data

The test split is touched exactly once, after model selection, for the
selected model. Research prototype only; not a diagnostic system.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from quantum_preprocessing import (  # noqa: E402
    CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, VOC_COLUMNS, PreprocessConfig,
    QuantumFeaturePreprocessor, encode_target, make_splits, make_synthetic_dataset, validate_dataframe,
)
from vqc_model import (  # noqa: E402
    QuantumPipeline, VQCClassifier, VQCConfig, compute_metrics, environment_info,
    generalization_gap, perturbation_sensitivity,
)

DISCLAIMER = ("Research prototype. Not a validated diagnostic system; good held-out performance "
              "does not establish clinical validity. Simulated on a classical computer; no quantum "
              "advantage is claimed.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--data", type=str, help="CSV/XLSX file with the expected columns")
    d.add_argument("--synthetic", action="store_true", help="use SYNTHETIC data (software testing only)")
    d.add_argument("--n-synthetic", type=int, default=1000)
    d.add_argument("--positive-label", type=str, default=None,
                   help="raw 'Cancer' value that means the positive class (required for real data; never assumed)")
    d.add_argument("--group-column", type=str, default=None, help="patient id column for patient-level splitting")
    d.add_argument("--output-dir", type=str, default="artifacts")
    m = p.add_argument_group("model")
    m.add_argument("--n-qubits", type=int, default=4)
    m.add_argument("--n-layers", type=int, default=2)
    m.add_argument("--reupload", action="store_true")
    m.add_argument("--entanglement", choices=["ring", "linear", "none"], default="ring")
    m.add_argument("--reduction", choices=["pca", "select_k_best", "none"], default="pca")
    m.add_argument("--optimizer", choices=["adam", "sgd", "adagrad", "rmsprop", "nesterov"], default="adam")
    m.add_argument("--learning-rate", type=float, default=0.05)
    m.add_argument("--epochs", type=int, default=40)
    m.add_argument("--batch-size", type=int, default=32)
    m.add_argument("--patience", type=int, default=8)
    m.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--n-runs", type=int, default=3, help="repeated trainings with different initialisations")
    e = p.add_argument_group("extra diagnostics")
    e.add_argument("--learning-curve", action="store_true", help="train on growing fractions of the training set")
    e.add_argument("--robustness", action="store_true", help="simulated VOC noise test on VALIDATION data")
    e.add_argument("--noise-levels", type=float, nargs="+", default=[0.01, 0.05, 0.10],
                   help="hypothetical relative noise levels (NOT measured sensor tolerances)")
    e.add_argument("--report-test-all-runs", action="store_true",
                   help="also report test metrics of every run (reporting only; never used for selection)")
    p.add_argument("--quiet", action="store_true")
    return p


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.synthetic:
        print("!! USING SYNTHETIC DATA: for software testing only; results have no scientific meaning !!")
        return make_synthetic_dataset(args.n_synthetic, seed=args.seed)
    if not args.data:
        raise SystemExit("Provide --data <file> or --synthetic")
    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    return pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)


def _mean_std(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    v = [x for x in values if x is not None]
    return {"mean": float(np.mean(v)), "std": float(np.std(v))} if v else {"mean": None, "std": None}


def _plot_losses(runs: List[Dict[str, Any]], selected_seed: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in runs:
        if r["status"].startswith("failed"):
            continue
        h, sel = r["history"], r["seed"] == selected_seed
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color="tab:blue", alpha=1.0 if sel else 0.25, lw=2 if sel else 1,
                label="train (selected)" if sel else None)
        ax.plot(ep, h["val_loss"], color="tab:orange", alpha=1.0 if sel else 0.25, lw=2 if sel else 1,
                label="validation (selected)" if sel else None)
        if sel:
            ax.axvline(r["best_epoch"], color="gray", ls="--", lw=1, label="best epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("weighted BCE loss"); ax.set_title("Training / validation loss")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_eval(y_test: np.ndarray, scores: np.ndarray, metrics: Dict[str, Any], out: Path) -> None:
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Test confusion matrix"); fig.tight_layout(); fig.savefig(out / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_test, scores)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate"); ax.legend()
        ax.set_title("Test ROC"); fig.tight_layout(); fig.savefig(out / "roc_test.png", dpi=150); plt.close(fig)


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    # ---- 1. load + validate -------------------------------------------------
    raw = load_data(args)
    df, vreport = validate_dataframe(raw, group_column=args.group_column)
    if args.synthetic:
        positive = 1
    elif args.positive_label is None:
        raise SystemExit(
            "--positive-label is required for real data (the meaning of the 'Cancer' codes is never assumed). "
            f"Observed target values: {list(vreport['target_distribution'])}. "
            f"Observed codes in Sex/Smoking_Status (treated as nominal categories): "
            f"{vreport['categorical_values_for_user_review']}"
        )
    else:
        positive = args.positive_label
    y = encode_target(df[TARGET_COLUMN], positive)
    groups = df[args.group_column].to_numpy() if args.group_column else None
    if verbose:
        print(f"Rows: {vreport['n_rows_clean']}  target distribution: {vreport['target_distribution']}")
        for w in vreport["warnings"]:
            print("  warning:", w)

    # ---- 2. split (agreed once, saved) -------------------------------------
    splits = make_splits(y, groups, seed=args.seed)
    (out / "split_indices.json").write_text(json.dumps({k: v.tolist() for k, v in splits.items()}))
    d_tr, d_va, d_te = (df.iloc[splits[k]] for k in ("train", "val", "test"))
    y_tr, y_va, y_te = (y[splits[k]] for k in ("train", "val", "test"))
    if verbose:
        print(f"Split sizes: train={len(d_tr)} val={len(d_va)} test={len(d_te)}")

    # ---- 3. preprocessing: fit on TRAIN only --------------------------------
    pre_cfg = PreprocessConfig(n_features=args.n_qubits, reduction=args.reduction, seed=args.seed)
    pre = QuantumFeaturePreprocessor(pre_cfg)
    X_tr = pre.fit_transform(d_tr, y_tr)
    X_va, X_te = pre.transform(d_va), pre.transform(d_te)
    if verbose and args.reduction == "pca":
        print(f"PCA keeps {pre.describe()['pca_total_explained_variance']:.1%} of encoded-feature variance")

    base_cfg = VQCConfig(
        n_qubits=args.n_qubits, n_layers=args.n_layers, reupload=args.reupload, entanglement=args.entanglement,
        optimizer=args.optimizer, learning_rate=args.learning_rate, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience, seed=args.seed,
        class_weight=None if args.class_weight == "none" else "balanced",
    )

    # ---- 4. repeated training runs (validation-based selection) -------------
    runs: List[Dict[str, Any]] = []
    models: Dict[int, VQCClassifier] = {}
    for r in range(args.n_runs):
        seed = args.seed + r
        cfg = VQCConfig(**{**asdict(base_cfg), "seed": seed})
        if verbose:
            print(f"\n=== run {r + 1}/{args.n_runs} (seed {seed}) ===")
        rec: Dict[str, Any] = {"seed": seed, "status": "ok", "error": None}
        try:
            model = VQCClassifier(cfg).fit(X_tr, y_tr, X_va, y_va, verbose=verbose)
            models[seed] = model
            info = model.training_info_
            rec.update(status=info["convergence_status"], best_epoch=info["best_epoch"],
                       epochs_completed=info["epochs_completed"], best_val_loss=info["best_monitored_loss"],
                       train_seconds=info["train_seconds"], threshold=model.threshold_,
                       history=model.history_,
                       train_metrics=compute_metrics(y_tr, model.predict_scores(X_tr), model.threshold_),
                       val_metrics=compute_metrics(y_va, model.predict_scores(X_va), model.threshold_))
            if args.report_test_all_runs:
                rec["test_metrics_report_only"] = compute_metrics(y_te, model.predict_scores(X_te), model.threshold_)
        except Exception as exc:  # recorded, never silently dropped
            rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"  run failed: {rec['error']}")
        runs.append(rec)

    ok = [r for r in runs if not r["status"].startswith("failed")]
    if not ok:
        raise SystemExit("All runs failed; see metrics.json for errors")
    sel = min(ok, key=lambda r: r["best_val_loss"])
    best = models[sel["seed"]]

    # ---- 5. final evaluation: test set touched once -------------------------
    thr = best.threshold_
    train_m = sel["train_metrics"]
    val_m = sel["val_metrics"]
    test_scores = best.predict_scores(X_te)
    test_m = compute_metrics(y_te, test_scores, thr)
    gap = {"train_minus_validation": generalization_gap(train_m, val_m),
           "validation_minus_test": generalization_gap(val_m, test_m),
           "definition": "gap = metric(first) - metric(second); positive means the first split scores higher. "
                         "Diagnostic only, not proof of model quality."}
    run_summary = {k: {"validation": _mean_std([r["val_metrics"][k] for r in ok]),
                       "train": _mean_std([r["train_metrics"][k] for r in ok])}
                   for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")}
    run_summary["best_val_loss"] = _mean_std([r["best_val_loss"] for r in ok])
    if args.report_test_all_runs:
        run_summary["test_report_only"] = {
            k: _mean_std([r["test_metrics_report_only"][k] for r in ok])
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")}

    # ---- 6. optional diagnostics (validation data only) ---------------------
    pipeline = QuantumPipeline(pre, best)
    extras: Dict[str, Any] = {}
    if args.robustness:
        extras["robustness_simulated_validation_only"] = perturbation_sensitivity(
            pipeline, d_va, y_va, VOC_COLUMNS, args.noise_levels, seed=args.seed)
    if args.learning_curve:
        lc = []
        for frac in (0.25, 0.5, 0.75, 1.0):
            if frac < 1.0:
                idx, _ = train_test_split(np.arange(len(y_tr)), train_size=frac, stratify=y_tr, random_state=args.seed)
            else:
                idx = np.arange(len(y_tr))
            m = VQCClassifier(base_cfg).fit(X_tr[idx], y_tr[idx], X_va, y_va, verbose=False)
            lc.append({"fraction": frac, "n_train": int(len(idx)),
                       "train_balanced_accuracy": compute_metrics(y_tr[idx], m.predict_scores(X_tr[idx]), m.threshold_)["balanced_accuracy"],
                       "val_balanced_accuracy": compute_metrics(y_va, m.predict_scores(X_va), m.threshold_)["balanced_accuracy"]})
        extras["learning_curve"] = lc
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot([r["n_train"] for r in lc], [r["train_balanced_accuracy"] for r in lc], "o-", label="train")
        ax.plot([r["n_train"] for r in lc], [r["val_balanced_accuracy"] for r in lc], "o-", label="validation")
        ax.set_xlabel("training samples"); ax.set_ylabel("balanced accuracy"); ax.legend()
        ax.set_title("Learning curve"); fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=150); plt.close(fig)

    # ---- 7. save everything --------------------------------------------------
    pipeline.save(out)
    _plot_losses(runs, sel["seed"], out / "loss_curves.png")
    _plot_eval(y_te, test_scores, test_m, out)
    t0 = time.perf_counter(); best.predict(X_te); inf_total = time.perf_counter() - t0
    results = {
        "disclaimer": DISCLAIMER,
        "data_source": "SYNTHETIC (software testing only)" if args.synthetic else str(args.data),
        "positive_label": str(positive),
        "validation_report": vreport,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "patient_level_split": bool(args.group_column),
        "config": asdict(base_cfg), "preprocessing": pre.describe(),
        "model_summary": {"n_qubits": base_cfg.n_qubits, "n_layers": base_cfg.n_layers,
                          "n_trainable_parameters": base_cfg.n_parameters,
                          "transformed_feature_dimension": args.n_qubits, "n_runs": args.n_runs},
        "runs": runs, "run_summary": run_summary, "selected_run_seed": sel["seed"],
        "selection_rule": "lowest best validation loss; threshold chosen on validation data",
        "train": train_m, "validation": val_m, "test": test_m, "generalization_gap": gap,
        "test_inference_seconds_total": float(inf_total),
        "n_failed_runs": len(runs) - len(ok), "extras": extras, "environment": environment_info(),
    }
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "config.json").write_text(json.dumps({"vqc": asdict(base_cfg), "preprocessing": asdict(pre_cfg),
                                                  "cli": vars(args)}, indent=2, default=str))

    # ---- 8. concise summary ---------------------------------------------------
    print("\n" + "=" * 64)
    print(f"Selected run seed={sel['seed']}  qubits={base_cfg.n_qubits} layers={base_cfg.n_layers} "
          f"params={base_cfg.n_parameters}  threshold={thr:.2f}")
    print(f"Runs: {len(ok)} ok, {len(runs) - len(ok)} failed;  "
          f"val balanced acc mean±std = {run_summary['balanced_accuracy']['validation']['mean']:.3f} "
          f"± {run_summary['balanced_accuracy']['validation']['std']:.3f}")
    for name, m in (("train", train_m), ("val", val_m), ("TEST", test_m)):
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        print(f"{name:5s} acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} "
              f"sens={m['recall_sensitivity']:.3f} spec={m['specificity']:.3f} macroF1={m['macro_f1']:.3f} AUC={auc}")
    print(f"gap(train-val) balanced_accuracy = {gap['train_minus_validation']['balanced_accuracy']:+.3f}")
    print(f"Artifacts written to: {out.resolve()}")
    print(DISCLAIMER)
    return results


if __name__ == "__main__":
    run()
