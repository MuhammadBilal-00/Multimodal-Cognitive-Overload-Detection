"""Clip-level evaluation + decision-threshold tuning + binary screening
(Part A of the honest-evaluation pass; see docs/results/honest_evaluation.md).

Why this exists: DAiSEE labels are PER-CLIP, and the published benchmark
numbers this project compares itself against (e.g. Abedi & Khan 2021's
63.9% accuracy) are per-clip accuracies -- but every metric this project
has reported so far is per-WINDOW (~8 overlapping windows per clip, each
scored independently). That is an apples-to-oranges comparison in both
directions: window-level scoring is noisier (a clip with 5/8 windows right
counts as 5 hits and 3 misses instead of one correct clip), and it was
never how the benchmark is defined. This script aggregates each clip's
window softmax probabilities (mean of probs -> argmax; majority vote as a
cross-check) and reports the clip-level metrics that are actually
comparable to the literature.

Three evaluations, all on Validation by default:
  1. Clip-level 4-class: accuracy / macro-F1 / QWK, plus the 3-class merge.
  2. Per-class decision-threshold tuning: coordinate ascent over 4
     per-class log-prob offsets, maximising CLIP-level macro-F1 on
     Validation. Decision-layer calibration only -- no retraining, never
     tuned on Test. Tuned and untuned both reported.
  3. Binary disengagement screening: classes {0,1} -> disengaged, {2,3} ->
     engaged, score = P(0)+P(1). Reported as accuracy PLUS balanced
     accuracy, ROC-AUC, sensitivity (recall of disengaged) and specificity
     -- accuracy alone is meaningless at ~95% engaged prevalence and must
     never be quoted without the companions.

Checkpoint bookkeeping (matters for the Test-consumed-once rule):
  - Validation default: artifacts/runs/20260815_181604 -- the checkpoint
    behind the SHIPPED model_int8.onnx and the committed
    metrics_validation.csv (states-head retrain, 2026-08-16).
  - Test (explicit --split Test only): artifacts/runs/20260801_185630 --
    the checkpoint whose Test predictions were frozen when Test was
    consumed exactly once (2026-08-02, committed metrics_test.csv).
    Re-aggregating those same frozen predictions per clip is a
    re-aggregation of existing outputs, not new model selection (same
    precedent as significance.py). Threshold-tuned variants are reported
    on Validation only.

Before any clip-level number is trusted, the script recomputes the
window-level metrics and asserts they match the committed
metrics_{split}.csv values -- if that cross-check fails, everything else
is suspect and the script aborts.

Usage:
  python ml/src/clip_eval.py                       # Validation
  python ml/src/clip_eval.py --split Test          # one-time final read
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score)

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from model import EngagementTCN  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
LABELS = [0, 1, 2, 3]
MERGE_3CLASS = {0: 0, 1: 0, 2: 1, 3: 2}  # same collapse as eval.py/baselines.py

DEFAULT_CHECKPOINTS = {
    # shipped model / committed metrics_validation.csv (macro-F1 0.3043)
    "Validation": RUNS_DIR / "20260815_181604" / "best.pt",
    # frozen consumed-once Test checkpoint (committed metrics_test.csv, 0.2475)
    "Test": RUNS_DIR / "20260801_185630" / "best.pt",
}
# committed window-level macro-F1 values the cross-check must reproduce
COMMITTED_WINDOW_MACRO_F1 = {"Validation": 0.3043, "Test": 0.2475}


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                          zero_division=0))


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def predict_probs(checkpoint: Path, x: np.ndarray) -> np.ndarray:
    model = EngagementTCN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True))
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            logits, _ = model(torch.from_numpy(x[i:i + 512]))
            probs.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(probs)


def aggregate_clips(probs: np.ndarray, y: np.ndarray,
                    clip_ids: np.ndarray) -> tuple:
    """Returns (clip_probs (C,4), clip_labels (C,), clip_majority_pred (C,)),
    ordered by first appearance. Clip label = the label its windows share
    (constant within a clip by dataset.py's construction, asserted here)."""
    order = {}
    for cid in clip_ids:
        if cid not in order:
            order[cid] = len(order)
    n_clips = len(order)
    clip_probs = np.zeros((n_clips, len(LABELS)))
    clip_labels = np.full(n_clips, -1, dtype=np.int64)
    clip_majority = np.zeros(n_clips, dtype=np.int64)
    for cid, idx in order.items():
        mask = clip_ids == cid
        clip_probs[idx] = probs[mask].mean(axis=0)
        labels_here = np.unique(y[mask])
        assert len(labels_here) == 1, f"clip {cid} has mixed labels"
        clip_labels[idx] = labels_here[0]
        clip_majority[idx] = np.bincount(probs[mask].argmax(1),
                                         minlength=4).argmax()
    return clip_probs, clip_labels, clip_majority


def tune_offsets(clip_probs: np.ndarray, clip_labels: np.ndarray,
                 rounds: int = 3) -> np.ndarray:
    """Coordinate ascent over per-class log-prob offsets, maximising
    clip-level macro-F1. Decision-layer calibration on the split given
    (Validation only, by the caller's discipline)."""
    log_probs = np.log(clip_probs + 1e-12)
    offsets = np.zeros(4)
    grid = np.arange(-2.0, 2.01, 0.1)
    best = macro_f1(clip_labels, (log_probs + offsets).argmax(1))
    for _ in range(rounds):
        for c in range(4):
            for cand in grid:
                trial = offsets.copy()
                trial[c] = cand
                score = macro_f1(clip_labels, (log_probs + trial).argmax(1))
                if score > best:
                    best, offsets = score, trial
    return offsets


def binary_screening(clip_probs: np.ndarray, clip_labels: np.ndarray) -> dict:
    """Disengaged = classes {0,1} (positive class -- the rare, important
    one), engaged = {2,3}. Score for AUC = P(0)+P(1)."""
    y_bin = (clip_labels <= 1).astype(int)          # 1 = disengaged
    score = clip_probs[:, 0] + clip_probs[:, 1]
    pred = (score >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    tn = int(((pred == 0) & (y_bin == 0)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    return {
        "prevalence_disengaged": round(float(y_bin.mean()), 4),
        "accuracy": round(float((pred == y_bin).mean()), 4),
        "balanced_accuracy": round(float(
            balanced_accuracy_score(y_bin, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_bin, score)), 4),
        "sensitivity_recall_disengaged": round(tp / (tp + fn), 4) if tp + fn else None,
        "specificity_recall_engaged": round(tn / (tn + fp), 4) if tn + fp else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def metrics_block(y_true, y_pred) -> dict:
    y3_true = np.vectorize(MERGE_3CLASS.get)(y_true)
    y3_pred = np.vectorize(MERGE_3CLASS.get)(y_pred)
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "qwk": round(qwk(y_true, y_pred), 4),
        "macro_f1_3class_merged": round(float(
            f1_score(y3_true, y3_pred, average="macro", zero_division=0)), 4),
        "accuracy_3class_merged": round(float((y3_true == y3_pred).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="Validation",
                        choices=["Validation", "Test"])
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="override the split's default checkpoint "
                             "(see module docstring for the bookkeeping)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.split]
    if args.split == "Test":
        print("NOTE: Test read -- re-aggregation of the frozen consumed-once "
              "checkpoint's predictions per clip. No threshold tuning is "
              "applied on Test (offsets, if reported, were tuned on "
              "Validation only).")

    data = np.load(DATASET_DIR / f"{args.split}.npz")
    x, y, clip_ids = data["x"], data["y_engagement"], data["clip_ids"]

    probs = predict_probs(checkpoint, x)

    # --- cross-check: window-level metrics must match the committed CSVs ---
    window_macro = round(macro_f1(y, probs.argmax(1)), 4)
    expected = COMMITTED_WINDOW_MACRO_F1[args.split]
    assert abs(window_macro - expected) < 5e-4, (
        f"window-level macro-F1 {window_macro} does not reproduce the "
        f"committed {expected} for {args.split} -- wrong checkpoint or "
        f"dataset drift; refusing to report clip-level numbers")
    print(f"cross-check OK: window-level macro-F1 {window_macro} matches "
          f"committed metrics_{args.split.lower()}.csv")

    clip_probs, clip_labels, clip_majority = aggregate_clips(probs, y, clip_ids)
    clip_pred = clip_probs.argmax(1)

    out = {
        "split": args.split,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "n_clips": int(len(clip_labels)),
        "n_windows": int(len(y)),
        "clip_label_counts": [int(c) for c in
                              np.bincount(clip_labels, minlength=4)],
        "window_level_4class": metrics_block(y, probs.argmax(1)),
        "clip_level_4class_meanprob": metrics_block(clip_labels, clip_pred),
        "clip_level_4class_majorityvote": metrics_block(clip_labels,
                                                        clip_majority),
        "clip_level_binary_screening": binary_screening(clip_probs,
                                                        clip_labels),
    }

    if args.split == "Validation":
        offsets = tune_offsets(clip_probs, clip_labels)
        log_probs = np.log(clip_probs + 1e-12)
        tuned_pred = (log_probs + offsets).argmax(1)
        out["clip_level_4class_threshold_tuned"] = metrics_block(clip_labels,
                                                                 tuned_pred)
        out["tuned_offsets"] = [round(float(o), 2) for o in offsets]
    else:
        # Test: apply the VALIDATION-frozen offsets unchanged (pre-registered
        # before this read; tuning itself never touches Test). Caveat carried
        # into the output: the offsets were tuned on Validation with the
        # shipped checkpoint (20260815_181604); Test uses the frozen original
        # checkpoint (20260801_185630) -- same architecture and
        # hyperparameters, different states-head training run.
        val_json = RESULTS_DIR / "clip_eval_validation.json"
        if val_json.exists():
            offsets = np.array(json.loads(val_json.read_text())["tuned_offsets"])
            log_probs = np.log(clip_probs + 1e-12)
            tuned_pred = (log_probs + offsets).argmax(1)
            out["clip_level_4class_valfrozen_offsets"] = metrics_block(
                clip_labels, tuned_pred)
            out["valfrozen_offsets"] = [round(float(o), 2) for o in offsets]

    out_path = RESULTS_DIR / f"clip_eval_{args.split.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {out_path}")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
