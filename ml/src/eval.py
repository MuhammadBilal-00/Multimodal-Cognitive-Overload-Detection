"""Evaluation artifacts (A8): confusion matrices, per-class metrics, ROC.

By default evaluates on VALIDATION. The test split is touched exactly once,
at the very end, via an explicit --split Test.

Also evaluates the quantized ONNX model on the same split when
--int8 <path> is given, producing the fp32-vs-int8 comparison row.

Outputs into docs/results/:
  confusion_{split}.png            raw + row-normalised, side by side
  metrics_{split}.csv              per-class P/R/F1 + macro/weighted + acc
  metrics_states_{split}.csv       secondary head: per-state P/R/F1 + AP/AUC
                                   against each state's base rate
  roc_{split}.png                  one-vs-rest ROC + AUC
  quantization.csv                 (with --int8) fp32 vs int8 macro-F1/size

Usage:
  python ml/src/eval.py --checkpoint artifacts/runs/<ts>/best.pt
  python ml/src/eval.py --checkpoint ... --split Test --int8 web/public/model/model_int8.onnx
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc, average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve)

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from labels import LABEL_COLS  # noqa: E402
from model import EngagementTCN  # noqa: E402

LABELS = ["very low (0)", "low (1)", "engaged (2)", "very engaged (3)"]
INK = "#333333"


def predict_torch(checkpoint: Path,
                  x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (engagement_logits, states_logits)."""
    model = EngagementTCN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True))
    model.eval()
    eng, states = [], []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            out_eng, out_states = model(torch.from_numpy(x[i:i + 512]))
            eng.append(out_eng.numpy())
            states.append(out_states.numpy())
    return np.concatenate(eng), np.concatenate(states)


def predict_onnx(onnx_path: Path,
                 x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (engagement_logits, states_logits)."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    eng, states = [], []
    for i in range(0, len(x), 512):
        out_eng, out_states = session.run(
            ["engagement", "states"], {"features": x[i:i + 512]})
        eng.append(out_eng)
        states.append(out_states)
    return np.concatenate(eng), np.concatenate(states)


def plot_confusion(y_true, y_pred, split: str, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, mat, title, fmt in (
            (axes[0], cm, "Counts", "d"),
            (axes[1], cm_norm, "Row-normalised", ".2f")):
        im = ax.imshow(mat, cmap="Blues",
                       vmax=mat.max() if title == "Counts" else 1.0)
        for r in range(4):
            for c in range(4):
                v = mat[r, c]
                colour = "white" if v > 0.6 * (mat.max() or 1) else INK
                ax.text(c, r, format(v, fmt), ha="center", va="center",
                        color=colour, fontsize=10)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(LABELS, rotation=20, ha="right", fontsize=8)
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_xlabel("Predicted", color=INK)
        ax.set_ylabel("True", color=INK)
        ax.set_title(f"{title} — {split}", color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{split.lower()}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, probs, split: str, out_dir: Path) -> dict:
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6"]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    aucs = {}
    for cls in range(4):
        fpr, tpr, _ = roc_curve((y_true == cls).astype(int), probs[:, cls])
        cls_auc = auc(fpr, tpr)
        aucs[LABELS[cls]] = float(cls_auc)
        ax.plot(fpr, tpr, color=colours[cls], linewidth=2,
                label=f"{LABELS[cls]} (AUC {cls_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate", color=INK)
    ax.set_ylabel("True positive rate", color=INK)
    ax.set_title(f"One-vs-rest ROC — {split}", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{split.lower()}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return aucs


def metrics_rows(y_true, y_pred) -> list[list]:
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    rows = [[LABELS[i], round(prec[i], 4), round(rec[i], 4),
             round(f1[i], 4), int(support[i])] for i in range(4)]
    rows.append(["macro", "", "",
                 round(f1_score(y_true, y_pred, average="macro",
                                zero_division=0), 4), len(y_true)])
    rows.append(["weighted", "", "",
                 round(f1_score(y_true, y_pred, average="weighted",
                                zero_division=0), 4), len(y_true)])
    rows.append(["accuracy", "", "",
                 round(float((y_true == y_pred).mean()), 4), len(y_true)])
    majority = int(np.bincount(y_true, minlength=4).argmax())
    rows.append(["majority baseline (macro-F1)", "", "",
                 round(f1_score(y_true, np.full_like(y_true, majority),
                                average="macro", zero_division=0), 4),
                 len(y_true)])
    # Ordinal-aware metric: engagement levels are ordered (very low < low <
    # engaged < very engaged), so quadratic-weighted kappa penalises a
    # far-off misclassification more than an adjacent one — macro-F1 above
    # does not distinguish the two. See docs/results/model_comparison_summary.md.
    rows.append(["quadratic weighted kappa", "", "",
                 round(float(cohen_kappa_score(y_true, y_pred,
                                               weights="quadratic")), 4),
                 len(y_true)])
    return rows


def states_rows(y_states: np.ndarray, states_logits: np.ndarray) -> list[list]:
    """Per-state metrics for the secondary multi-label head.

    Until now this head was never scored anywhere in the project (train.py
    discards the state targets in its eval loop; this file only ever looked at
    `engagement`). It is trained with an UNWEIGHTED BCEWithLogitsLoss over very
    imbalanced targets, so the thing that actually needs checking is whether it
    learned anything beyond the base rate.

    That is what the `prevalence` and `pred_rate` columns are for:
      * pred_rate ~= 1.0 or ~= 0.0 while prevalence sits in between
        => the head is just predicting the majority answer.
      * average_precision ~= prevalence and roc_auc ~= 0.5
        => no discriminative signal at all, regardless of how good the
           accuracy column looks (accuracy is meaningless at 95% prevalence).
    """
    probs = 1.0 / (1.0 + np.exp(-states_logits))
    pred = (probs >= 0.5).astype(int)
    rows = []
    for i, name in enumerate(LABEL_COLS):
        true_i = y_states[:, i].astype(int)
        pred_i = pred[:, i]
        prob_i = probs[:, i]
        prevalence = float(true_i.mean())
        prec, rec, f1, _ = precision_recall_fscore_support(
            true_i, pred_i, average="binary", zero_division=0)
        # AUC/AP are undefined for a single-class column; emit "" not a crash.
        if true_i.min() == true_i.max():
            ap = auc_score = ""
        else:
            ap = round(float(average_precision_score(true_i, prob_i)), 4)
            auc_score = round(float(roc_auc_score(true_i, prob_i)), 4)
        rows.append([
            name.lower(), i,
            round(float(prec), 4), round(float(rec), 4), round(float(f1), 4),
            ap, auc_score,
            round(prevalence, 4), round(float(pred_i.mean()), 4),
            round(float((true_i == pred_i).mean()), 4),
            int(true_i.sum()), len(true_i),
        ])
    macro_f1 = float(np.mean([r[4] for r in rows]))
    rows.append(["macro", "", "", "", round(macro_f1, 4), "", "", "", "", "",
                 "", len(y_states)])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="Validation",
                        choices=["Validation", "Test"])
    parser.add_argument("--int8", type=Path, default=None,
                        help="also evaluate this ONNX model (quantization row)")
    parser.add_argument("--dataset-dir", type=Path,
                        default=REPO_ROOT / "artifacts" / "dataset",
                        help="directory holding {split}.npz")
    # Not just convenience: without it any smoke run overwrites the committed
    # real metrics/figures in docs/results/ with throwaway numbers.
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "docs" / "results",
                        help="where to write metrics/figures")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.dataset_dir / f"{args.split}.npz"
    if not npz_path.exists():
        raise SystemExit(
            f"{npz_path} not found — build it with `python ml/src/dataset.py` "
            f"first (needs the DAiSEE dataset), or point --dataset-dir at an "
            f"existing bundle.")
    data = np.load(npz_path)
    x = data["x"]
    y = data["y_engagement"]

    logits, states_logits = predict_torch(args.checkpoint, x)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = logits.argmax(axis=1)

    plot_confusion(y, y_pred, args.split, out_dir)
    aucs = plot_roc(y, probs, args.split, out_dir)

    rows = metrics_rows(y, y_pred)
    # secondary result sanctioned by the brief: merge levels 0+1 as "low"
    merge = {0: 0, 1: 0, 2: 1, 3: 2}
    y3 = np.vectorize(merge.get)(y)
    p3 = np.vectorize(merge.get)(y_pred)
    rows.append(["3-class merged (0+1=low) macro-F1", "", "",
                 round(f1_score(y3, p3, average="macro", zero_division=0), 4),
                 len(y)])

    with open(out_dir / f"metrics_{args.split.lower()}.csv", "w",
              newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        writer.writerows(rows)

    # Secondary multi-label head. Channel order is LABEL_COLS (Boredom,
    # Engagement, Confusion, Frustration) — the browser mirrors it in
    # web/lib/states.ts, and CONTRACT.md §5 documents it.
    s_rows = states_rows(data["y_states"], states_logits)
    with open(out_dir / f"metrics_states_{args.split.lower()}.csv", "w",
              newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["state", "channel_index", "precision", "recall", "f1",
                         "average_precision", "roc_auc", "prevalence",
                         "pred_rate", "accuracy", "positives", "support"])
        writer.writerows(s_rows)

    fp32_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    print(f"{args.split}: fp32 macro-F1 {fp32_macro:.4f}")
    print(json.dumps({"auc": aucs}, indent=1))
    print(f"states macro-F1 {s_rows[-1][4]} "
          f"-> {out_dir / f'metrics_states_{args.split.lower()}.csv'}")

    if args.int8 is not None:
        int8_logits, _ = predict_onnx(args.int8, x)
        int8_pred = int8_logits.argmax(axis=1)
        int8_macro = f1_score(y, int8_pred, average="macro", zero_division=0)
        fp32_size = (REPO_ROOT / "artifacts" / "export"
                     / "model_fp32.onnx")
        with open(out_dir / "quantization.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["model", "size_bytes", f"macro_f1_{args.split}"])
            writer.writerow(["fp32",
                             fp32_size.stat().st_size if fp32_size.exists()
                             else "", round(float(fp32_macro), 4)])
            writer.writerow(["int8-QDQ", args.int8.stat().st_size,
                             round(float(int8_macro), 4)])
        print(f"int8 macro-F1 {int8_macro:.4f} "
              f"(delta {int8_macro - fp32_macro:+.4f})")


if __name__ == "__main__":
    main()
