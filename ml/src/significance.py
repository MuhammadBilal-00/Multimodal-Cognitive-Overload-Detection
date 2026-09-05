"""Clip-level bootstrap significance testing: is the TCN meaningfully
different from the best classical baseline, or is the Test-split crossover
(macro-F1 0.2475 TCN vs 0.2643 random forest, docs/results/baselines.csv)
noise from a small effective sample?

Windows are NOT independent samples: a 30-frame window at training stride
10 shares 20 of its 30 frames with its neighbour, so all windows drawn
from one clip are correlated. Treating "14,241 test windows" as 14,241
independent samples overstates the effective sample size behind any
macro-F1 comparison. This script instead resamples CLIP IDs (the
`clip_ids` array `dataset.py` already saves into every split's .npz, unused
by any other script) with replacement -- a paired cluster bootstrap -- so
the resampling unit matches the actual unit of independence in the data.

Reads only already-frozen artefacts: the TCN checkpoint that produced the
committed `docs/results/metrics_test.csv` (Test consumed exactly once,
2026-08-02, before the later states-head retrain),
plus classical baselines trained identically to baselines.py's (not
persisted there, so retrained here with the same hyperparameters/seed --
the Validation-split point estimates below should match baselines.csv
almost exactly as a build-in consistency check). This is a statistical
characterisation of existing numbers, not new model selection on Test.

Usage: python ml/src/significance.py [--iterations 2000] [--seed 42]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from baselines import aggregate_features  # noqa: E402
from model import EngagementTCN  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
LABELS = [0, 1, 2, 3]
# The checkpoint that produced the committed docs/results/metrics_test.csv
# (Test consumed exactly once, 2026-08-02) -- fixed here so this script's
# numbers stay directly comparable to that committed artefact rather than
# the later states-retrained checkpoint, which was never evaluated on Test.
TCN_CHECKPOINT = REPO_ROOT / "artifacts" / "runs" / "20260801_185630" / "best.pt"


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                          zero_division=0))


def load_npz(split: str) -> dict:
    data = np.load(DATASET_DIR / f"{split}.npz")
    return {"x": data["x"], "y": data["y_engagement"], "clip_ids": data["clip_ids"]}


def tcn_predict(x: np.ndarray) -> np.ndarray:
    model = EngagementTCN()
    model.load_state_dict(torch.load(TCN_CHECKPOINT, map_location="cpu",
                                     weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            logits, _ = model(torch.from_numpy(x[i:i + 512]))
            preds.append(logits.argmax(dim=1).numpy())
    return np.concatenate(preds)


def clip_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                   clip_ids: np.ndarray, iterations: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Paired cluster bootstrap over clip IDs. Returns (f1s_a, f1s_b)."""
    unique_clips = np.unique(clip_ids)
    windows_per_clip = [np.flatnonzero(clip_ids == c) for c in unique_clips]
    n_clips = len(unique_clips)
    f1s_a = np.empty(iterations)
    f1s_b = np.empty(iterations)
    for i in range(iterations):
        chosen = rng.integers(0, n_clips, size=n_clips)
        idx = np.concatenate([windows_per_clip[j] for j in chosen])
        f1s_a[i] = macro_f1(y_true[idx], pred_a[idx])
        f1s_b[i] = macro_f1(y_true[idx], pred_b[idx])
    return f1s_a, f1s_b


def ci95(values: np.ndarray) -> list:
    return [round(float(np.percentile(values, 2.5)), 4),
            round(float(np.percentile(values, 97.5)), 4)]


def evaluate_split(split: str, iterations: int, seed: int) -> dict:
    data = load_npz(split)
    x, y, clip_ids = data["x"], data["y"], data["clip_ids"]

    tcn_pred = tcn_predict(x)
    tcn_point = macro_f1(y, tcn_pred)

    train = load_npz("Train")
    x_train_agg = aggregate_features(train["x"])
    y_train = train["y"]
    x_agg = aggregate_features(x)

    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf.fit(x_train_agg, y_train)
    gbm = HistGradientBoostingClassifier(random_state=42)
    gbm.fit(x_train_agg, y_train,
            sample_weight=compute_sample_weight("balanced", y_train))

    rf_pred = rf.predict(x_agg)
    gbm_pred = gbm.predict(x_agg)
    rf_point = macro_f1(y, rf_pred)
    gbm_point = macro_f1(y, gbm_pred)
    baseline_name, baseline_pred, baseline_point = (
        ("random_forest", rf_pred, rf_point) if rf_point >= gbm_point
        else ("gradient_boosting", gbm_pred, gbm_point))

    rng = np.random.default_rng(seed)
    f1s_tcn, f1s_base = clip_bootstrap(y, tcn_pred, baseline_pred, clip_ids,
                                       iterations, rng)
    diff = f1s_tcn - f1s_base
    # Two-sided empirical p-value: how often the bootstrap disagrees with
    # the point estimate's sign (standard percentile-bootstrap hypothesis test).
    p_value = min(float(2 * min((diff <= 0).mean(), (diff >= 0).mean())), 1.0)

    return {
        "n_clips": int(len(np.unique(clip_ids))),
        "n_windows": int(len(y)),
        "tcn": {"point_macro_f1": round(tcn_point, 4), "ci_95": ci95(f1s_tcn)},
        "best_classical_baseline": {
            "model": baseline_name,
            "point_macro_f1": round(baseline_point, 4),
            "ci_95": ci95(f1s_base),
            "other_classical_point_macro_f1": {
                "random_forest": round(rf_point, 4),
                "gradient_boosting": round(gbm_point, 4),
            },
        },
        "tcn_minus_baseline": {
            "point": round(tcn_point - baseline_point, 4),
            "ci_95": ci95(diff),
        },
        "two_sided_p_value": round(p_value, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = {
        split: evaluate_split(split, args.iterations, args.seed)
        for split in ("Validation", "Test")
    }
    results["iterations"] = args.iterations
    results["method"] = ("paired cluster bootstrap, resampling unit = clip ID "
                         "(not window) -- see module docstring")
    results["tcn_checkpoint"] = str(TCN_CHECKPOINT.relative_to(REPO_ROOT))

    out_path = REPO_ROOT / "docs" / "results" / "significance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {out_path}")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
