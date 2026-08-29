"""CV-based feature selection (Phase 1 of the rigorous-fix pass): which
feature-family preset should the TCN actually be trained on? Runs each of
feature_groups.py's 4 presets through all 5 CV folds (cv_splits.py), using
cv_train.py's lean training loop at default hyperparameters -- Phase 2
tunes hyperparameters AFTER this selects the feature set. This ordering
is the point: it fixes the "feature selection happened after model
design" gap by making feature choice part of the search itself, on
resampled folds, rather than a single-split retrospective check.

Writes each (preset, fold) result to disk IMMEDIATELY after computing it
(append + flush), and skips combos already present on startup -- this
script trains in-process (unlike feature_ablation.py/multi_seed_
robustness.py, which shell out to train.py and get a persisted run
directory per config "for free"), so without this it has no recovery
story if interrupted mid-run, which this environment does periodically
over long background jobs.

Usage: python -u ml/src/cv_feature_selection.py   (unbuffered -- see above)
"""

import csv
import statistics
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from cv_splits import get_folds  # noqa: E402
from cv_train import train_and_evaluate  # noqa: E402
from feature_groups import PRESETS  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
OUT_PATH = REPO_ROOT / "docs" / "results" / "cv_feature_selection.csv"
HEADER = ["preset", "n_features", "fold", "macro_f1", "accuracy", "qwk"]


def load_done() -> set:
    """(preset, fold) pairs already completed, from a prior interrupted run."""
    if not OUT_PATH.exists():
        return set()
    with open(OUT_PATH, newline="") as fh:
        return {(row["preset"], int(row["fold"]))
                for row in csv.DictReader(fh)}


def append_row(row: list) -> None:
    is_new = not OUT_PATH.exists()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(HEADER)
        writer.writerow(row)
        fh.flush()


def main() -> None:
    data = np.load(DATASET_DIR / "Train.npz")
    x_full, y, ys, clip_ids = (data["x"], data["y_engagement"],
                               data["y_states"], data["clip_ids"])
    done = load_done()
    if done:
        print(f"resuming: {len(done)} (preset, fold) combos already done")

    for preset, feature_idx in PRESETS.items():
        x = x_full[:, :, feature_idx]
        for fold_idx, (train_idx, val_idx) in enumerate(get_folds(clip_ids, y)):
            if (preset, fold_idx) in done:
                print(f"preset={preset} fold={fold_idx} already done, skipping")
                continue
            print(f"preset={preset} fold={fold_idx} training...", flush=True)
            metrics = train_and_evaluate(
                x[train_idx], y[train_idx], ys[train_idx],
                x[val_idx], y[val_idx], ys[val_idx],
                hp={}, n_features=len(feature_idx))
            append_row([preset, len(feature_idx), fold_idx,
                       round(metrics["macro_f1"], 4),
                       round(metrics["accuracy"], 4),
                       round(metrics["qwk"], 4)])
            print(f"  macro_f1={metrics['macro_f1']:.4f}", flush=True)

    with open(OUT_PATH, newline="") as fh:
        by_preset = {}
        for row in csv.DictReader(fh):
            by_preset.setdefault(row["preset"], []).append(float(row["macro_f1"]))

    summary_rows = [["preset", "n_features", "mean_macro_f1", "std_macro_f1",
                     "min", "max"]]
    for preset, feature_idx in PRESETS.items():
        f1s = by_preset[preset]
        mean, std = statistics.mean(f1s), statistics.stdev(f1s)
        summary_rows.append([preset, len(feature_idx), round(mean, 4),
                            round(std, 4), round(min(f1s), 4), round(max(f1s), 4)])
        print(f"{preset}: mean={mean:.4f} std={std:.4f}")

    summary_path = REPO_ROOT / "docs" / "results" / "cv_feature_selection_summary.csv"
    with open(summary_path, "w", newline="") as fh:
        csv.writer(fh).writerows(summary_rows)
    print(f"wrote {OUT_PATH} and {summary_path}")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
