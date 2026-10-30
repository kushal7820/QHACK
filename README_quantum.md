# Quantum-Only VQC for Breath VOC Pattern Analysis

**Research prototype. Not a medical diagnostic system.** Good test scores do not
establish clinical validity; nothing here has been validated on patients or real
sensors. The model is simulated on a classical computer (PennyLane
`default.qubit`); no quantum speed-up or advantage is claimed.

The only predictive model is a trainable Variational Quantum Classifier (VQC).
There is no classical classifier and no classical-vs-quantum comparison.
Classical code is used only for preprocessing, optimisation and evaluation.

## Files

| File | Purpose |
|---|---|
| `quantum_preprocessing.py` | validation, patient-aware split, fit-on-train feature preparation, synthetic test data |
| `vqc_model.py` | the VQC (`VQCClassifier`), `QuantumPipeline`, metrics, perturbation test |
| `train_vqc.py` | CLI: train, select, evaluate, save |
| `tests/test_quantum_vqc.py` | pytest suite |
| `requirements_quantum.txt` | pinned dependencies |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements_quantum.txt pytest
pytest -q tests/
```

## Dataset requirements

CSV/XLSX with columns: `Age, Sex, Smoking_Status, Acetone, Isoprene, Ethanol,
Formaldehyde, Benzene, Toluene, Methane, Cancer`. `Cancer` is the binary target
and is never an input.

* The meaning of the codes is **not assumed**. `Sex` and `Smoking_Status` are
  one-hot encoded as nominal categories (no ordering imposed). You must tell the
  script which raw `Cancer` value is the positive class with `--positive-label`;
  without it the script stops and lists the observed codes.
* Validation reports: missing columns, non-numeric entries, missing values,
  duplicates (reported, not removed), negative VOC values and ages outside
  0-120 (set to missing, then imputed from training statistics), target balance.
* If one patient has several rows, pass `--group-column <patient_id>` so whole
  patients stay in one split. Without it rows are assumed independent.
* If no real data is available use `--synthetic` (software testing only; the
  numbers mean nothing).

## Run

```bash
python train_vqc.py --data breath_voc.csv --positive-label 1          # defaults: 4 qubits, 2 layers, 3 runs
python train_vqc.py --data breath_voc.csv --positive-label 1 --group-column PatientID \
    --n-qubits 4 --n-layers 3 --reupload --epochs 60 --patience 10 --n-runs 5 \
    --learning-curve --robustness --noise-levels 0.02 0.05
python train_vqc.py --synthetic                                       # smoke test
```

Key options: `--n-qubits --n-layers --reupload --entanglement {ring,linear,none}
--reduction {pca,select_k_best,none} --optimizer {adam,sgd,adagrad,rmsprop,nesterov}
--learning-rate --epochs --batch-size --patience --class-weight {balanced,none}
--seed --n-runs`.

## Feature preparation and qubits

Original features (Age + 7 VOCs numeric, Sex and Smoking_Status categorical)
become roughly 13 encoded columns after median imputation, standardisation and
one-hot encoding. These are reduced to **`n_qubits` dimensions** by PCA (or
SelectKBest), then min-max scaled to **[-pi, pi]** and clipped to that range
(unseen values outside the training range saturate rather than wrap around).
Each of those dimensions is one rotation angle. Original features therefore do
*not* map one-to-one to qubits; `metrics.json -> preprocessing` lists PCA
explained variance and loadings. Every fitted object (imputers, scaler,
encoder, reducer, angle scaler, column order) is fitted on the training split
only and saved.

## Circuit

For each layer: `RY(x_i)` on each qubit (first layer only, or every layer with
`--reupload`), then `Rot(phi, theta, omega)` on each qubit, then CNOTs (ring by
default). Output `z = <Z_0>` in [-1, 1]. Trainable parameters = `layers x qubits x 3`
(24 for the default). Score `p = (1+z)/2` is a bounded model score, **not** a
calibrated probability. Loss: class-weighted binary cross-entropy with
probabilities clipped to [1e-7, 1-1e-7]. Gradients: PennyLane backprop on the
simulator with batched (broadcast) circuit evaluation. Optimiser: Adam by default.

## Generalisation methodology

* One stratified 70/15/15 split (patient-grouped if requested), saved to
  `split_indices.json`.
* Validation set: early stopping (patience, best-parameter restoration), run
  selection (lowest best validation loss) and decision-threshold choice
  (maximum validation balanced accuracy, grid 0.05-0.95).
* Test set: evaluated once, for the selected model only, after selection.
  `--report-test-all-runs` additionally reports test metrics of every run, for
  reporting only; do not use them to choose between runs.
* Metrics: accuracy, balanced accuracy, precision, recall/sensitivity,
  specificity, macro-F1, ROC-AUC (only if both classes are present), confusion
  matrix (rows true 0/1, columns predicted 0/1).
* Generalisation gap = metric(train) - metric(validation) (and validation - test);
  positive means the earlier split scores higher. It is a diagnostic, not proof
  of quality (a small gap can also mean underfitting).
* Repeated runs: `--n-runs N` uses seeds `seed ... seed+N-1`; mean/std across
  runs are reported, failed runs are recorded in `metrics.json`, not dropped.
* Robustness (`--robustness`): simulated multiplicative Gaussian noise on VOC
  columns of the **validation** set at levels *you* choose (not measured sensor
  tolerances); labels untouched. This is not real sensor validation.
* Learning curve (`--learning-curve`): retrains on 25/50/75/100% of training data.

## Outputs (in `--output-dir`, default `artifacts/`)

`metrics.json` (all metrics, runtimes, convergence, versions, validation report),
`config.json`, `split_indices.json`, `loss_curves.png`, `roc_test.png`,
`confusion_matrix_test.png`, `learning_curve.png` (optional),
`model/vqc_params.npz` + `model/vqc_config.json`, `preprocessing/preprocessor.joblib`
+ `preprocessor_config.json`. The circuit is rebuilt from the saved config; no
live quantum node is serialised.

## Inference

```python
import pandas as pd
from vqc_model import QuantumPipeline
pipe = QuantumPipeline.load("artifacts")
df = pd.read_csv("new_samples.csv")            # same raw columns, any column order
scores = pipe.predict_scores(df)               # in [0, 1], not calibrated
labels = pipe.predict(df)                      # uses the saved validation-chosen threshold
```
This API is Streamlit-ready: load once (e.g. `st.cache_resource`), pass a one-row DataFrame.
Saved preprocessors are scikit-learn objects stored with joblib: load only
artifacts you trust and with the pinned versions.

## Computational limits

State-vector simulation scales as 2^n_qubits. 2-6 qubits are trivial; each
extra qubit doubles memory/time. Gradient cost with backprop grows with
circuit depth. Reference timing (this repo's synthetic run, 700 training rows,
4 qubits, 2 layers, CPU): roughly 0.7 s per epoch.

## Troubleshooting

* `ImportError` / API errors after upgrading PennyLane or NumPy: reinstall the pinned versions.
* `--positive-label ... not found`: use a value printed in the error message (compare as raw codes).
* `n_features exceeds encoded dimension`: lower `--n-qubits` (PCA cannot exceed ~13 here).
* Loss is NaN / run marked failed: lower `--learning-rate`; failed runs are listed in `metrics.json`.
* Predictions all one class: check class balance, keep `--class-weight balanced`, inspect `loss_curves.png`.
* Slow training: fewer qubits/layers, larger `--batch-size`, fewer `--n-runs`.

## Test status

`pytest -q tests/` covers circuit construction, output validity, parameter updates,
loss tracking, early stopping/restoration, preprocessing leakage and column order,
patient-level split, unseen-sample prediction, save/load equality, reproducibility
and an end-to-end run. 12 tests passed on Python 3.12 with the pinned versions on
synthetic data only. Not verified: behaviour on the real dataset, other
OS/Python versions, `select_k_best`/`reduction=none` paths, group splitting
inside `train_vqc.py` (the split function itself is tested), XLSX loading.
