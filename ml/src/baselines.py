"""Classical baselines for the engagement task (A5).

Flattens each (30, 13) window to a 65-dim aggregate vector (mean, std,
min, max, range per feature, in that order) and trains logistic
regression + random forest once on Train. Reports macro-F1 and accuracy
on BOTH Validation and Test splits — same two splits, same `split` column
convention as `eval.py`'s `metrics_validation.csv` / `metrics_test.csv`,
so the TCN and these baselines are directly comparable row-by-row.

Also reports the same "3-class merged (0+1=low)" collapse `eval.py`
reports for the TCN, as extra `*_3class_merged` model rows, for a cleaner
side-by-side thesis table.

Usage: python ml/src/baselines.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
MERGE_3CLASS = {0: 0, 1: 0, 2: 1, 3: 2}  # 0+1 -> "low", per eval.py


def aggregate_features(x: np.ndarray) -> np.ndarray:
    """(N, 30, 13) windows -> (N, 65): mean/std/min/max/range per feature."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    lo = x.min(axis=1)
    hi = x.max(axis=1)
    rng = hi - lo
    return np.concatenate([mean, std, lo, hi, rng], axis=1).astype(np.float32)


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATASET_DIR / f"{split}.npz")
    return aggregate_features(data["x"]), data["y_engagement"]


def merge_3class(y: np.ndarray) -> np.ndarray:
    return np.vectorize(MERGE_3CLASS.get)(y)


def row(split: str, model: str, y_true: np.ndarray, y_pred: np.ndarray) -> list:
    return [
        split, model,
        round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        round(float(accuracy_score(y_true, y_pred)), 4),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default is Validation-only: Test was already consumed once (2026-08-02)
    # for the TCN's final report, and eval.py requires an explicit --split
    # Test opt-in for the same reason. --split both matches how the existing
    # committed docs/results/baselines.csv was produced (2026-08-09) — that
    # artifact is not touched by this flag, only future re-runs are.
    parser.add_argument("--split", choices=["validation", "both"],
                        default="validation")
    args = parser.parse_args()

    x_train, y_train = load_split("Train")
    x_val, y_val = load_split("Validation")

    # Same idiom as train.py:233-237 / eval.py:134-138: majority-class
    # constant prediction, macro-F1 against it as the floor every model
    # must beat.
    majority = int(np.bincount(y_train, minlength=4).argmax())

    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)
    logreg.fit(x_train, y_train)
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf.fit(x_train, y_train)

    splits = [("Validation", x_val, y_val)]
    if args.split == "both":
        splits.append(("Test", *load_split("Test")))

    rows = [["split", "model", "macro_f1", "accuracy"]]
    for split_name, x_eval, y_eval in splits:
        maj_pred = np.full_like(y_eval, majority)
        logreg_pred = logreg.predict(x_eval)
        rf_pred = rf.predict(x_eval)

        rows.append(row(split_name, "majority", y_eval, maj_pred))
        rows.append(row(split_name, "logreg", y_eval, logreg_pred))
        rows.append(row(split_name, "random_forest", y_eval, rf_pred))

        y_eval_3 = merge_3class(y_eval)
        rows.append(row(split_name, "majority_3class_merged", y_eval_3, merge_3class(maj_pred)))
        rows.append(row(split_name, "logreg_3class_merged", y_eval_3, merge_3class(logreg_pred)))
        rows.append(row(split_name, "random_forest_3class_merged", y_eval_3, merge_3class(rf_pred)))

    out_dir = REPO_ROOT / "docs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baselines.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    print(f"wrote {out_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
