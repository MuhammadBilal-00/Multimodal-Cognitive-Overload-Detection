"""Ensemble fix (Part A4 of the honest-evaluation pass), implementing the
two remedies the committed diagnosis pointed at (docs/results/
rigorous_model_search.md, "Root cause, diagnosed directly"): the original
stack's unweighted meta-learner over-trusted the random forest's
majority-class overconfidence (never predicting classes 0/1 at all), and a
fully `class_weight="balanced"` meta-learner over-corrected into
instability. Remedies swept here:

  1. Drop RF from the stack (TCN+GBM probability columns only), and/or
  2. a MODERATE meta-learner class weighting: weight per class
     proportional to (1/frequency)^p for p in {0, 0.25, 0.5, 0.75, 1.0}
     (p=0 -> unweighted, p=1.0 -> equivalent direction to "balanced").

Full grid: {TCN+GBM, TCN+GBM+RF} x 5 weight powers = 10 configs, each
seconds to fit (meta-learner only -- reuses the cached OOF matrix from
ensemble_stack.py and the cached Validation base-learner probabilities
from the diagnosis pass; no base-learner retraining). Every config scored
on Validation at both window level and clip level (mean-prob aggregation,
same as clip_eval.py). Selection by clip-level macro-F1.

Usage: python ml/src/ensemble_fix.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
LABELS = [0, 1, 2, 3]

# column blocks in the cached OOF matrix (ensemble_stack.py's layout)
TCN_COLS = [0, 1, 2, 3]
RF_COLS = [4, 5, 6, 7]
GBM_COLS = [8, 9, 10, 11]

STACKS = {
    "tcn+gbm": TCN_COLS + GBM_COLS,
    "tcn+gbm+rf": TCN_COLS + RF_COLS + GBM_COLS,
}
WEIGHT_POWERS = [0.0, 0.25, 0.5, 0.75, 1.0]


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                          zero_division=0))


def class_weights_for_power(y: np.ndarray, power: float) -> dict | None:
    if power == 0.0:
        return None
    counts = np.bincount(y, minlength=4).astype(float)
    w = (len(y) / (4 * counts.clip(min=1))) ** power
    return {c: float(w[c]) for c in range(4)}


def clip_aggregate(pred_probs: np.ndarray, y: np.ndarray,
                   clip_ids: np.ndarray) -> tuple:
    order = {}
    for cid in clip_ids:
        if cid not in order:
            order[cid] = len(order)
    clip_probs = np.zeros((len(order), 4))
    clip_labels = np.full(len(order), -1, dtype=np.int64)
    for cid, idx in order.items():
        mask = clip_ids == cid
        clip_probs[idx] = pred_probs[mask].mean(axis=0)
        clip_labels[idx] = y[mask][0]
    return clip_probs.argmax(1), clip_labels


def main() -> None:
    train = np.load(DATASET_DIR / "Train.npz")
    y_train = train["y_engagement"]
    val = np.load(DATASET_DIR / "Validation.npz")
    y_val, clip_ids_val = val["y_engagement"], val["clip_ids"]

    oof = np.load(RUNS_DIR / "ensemble_oof_cache.npz")["oof"]
    diag = np.load(RUNS_DIR / "ensemble_diag_val.npz")
    val_blocks = np.concatenate(
        [diag["tcn_probs"], diag["rf_probs"], diag["gbm_probs"]], axis=1)

    rows = [["stack", "weight_power", "val_window_macro_f1",
             "val_window_accuracy", "val_clip_macro_f1", "val_clip_accuracy"]]
    for stack_name, cols in STACKS.items():
        for power in WEIGHT_POWERS:
            meta = LogisticRegression(
                max_iter=2000,
                class_weight=class_weights_for_power(y_train, power))
            meta.fit(oof[:, cols], y_train)
            val_probs = meta.predict_proba(val_blocks[:, cols])
            window_pred = val_probs.argmax(1)
            clip_pred, clip_labels = clip_aggregate(val_probs, y_val,
                                                    clip_ids_val)
            rows.append([
                stack_name, power,
                round(macro_f1(y_val, window_pred), 4),
                round(float((y_val == window_pred).mean()), 4),
                round(macro_f1(clip_labels, clip_pred), 4),
                round(float((clip_labels == clip_pred).mean()), 4),
            ])
            print(rows[-1])

    out_path = RESULTS_DIR / "ensemble_fixed.csv"
    with open(out_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"wrote {out_path}")

    best = max(rows[1:], key=lambda r: r[4])
    print(f"best by clip-level macro-F1: stack={best[0]} power={best[1]} "
          f"clip_macro_f1={best[4]} clip_acc={best[5]}")
    print("references -- tuned TCN alone (Validation): window macro-F1 "
          "0.3081; shipped TCN clip-level macro-F1 0.3099 "
          "(clip_eval_validation.json)")


if __name__ == "__main__":
    main()
