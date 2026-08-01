"""Evaluation artifacts (A8): confusion matrices, per-class metrics, ROC.

By default evaluates on VALIDATION. The test split is touched exactly once,
at the very end, via an explicit --split Test.

Also evaluates the quantized ONNX model on the same split when
--int8 <path> is given, producing the fp32-vs-int8 comparison row.

Outputs into docs/results/:
  confusion_{split}.png            raw + row-normalised, side by side
  metrics_{split}.csv              per-class P/R/F1 + macro/weighted + acc
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
    auc, confusion_matrix, f1_score, precision_recall_fscore_support,
    roc_curve)

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from model import EngagementTCN  # noqa: E402

LABELS = ["very low (0)", "low (1)", "engaged (2)", "very engaged (3)"]
INK = "#333333"


def predict_torch(checkpoint: Path, x: np.ndarray) -> np.ndarray:
    model = EngagementTCN()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True))
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            out, _ = model(torch.from_numpy(x[i:i + 512]))
            logits.append(out.numpy())
    return np.concatenate(logits)


def predict_onnx(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path),
                                   providers=["CPUExecutionProvider"])
    logits = []
    for i in range(0, len(x), 512):
        out = session.run(["engagement"], {"features": x[i:i + 512]})[0]
        logits.append(out)
    return np.concatenate(logits)


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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="Validation",
                        choices=["Validation", "Test"])
    parser.add_argument("--int8", type=Path, default=None,
                        help="also evaluate this ONNX model (quantization row)")
    args = parser.parse_args()

    out_dir = REPO_ROOT / "docs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(REPO_ROOT / "artifacts" / "dataset" / f"{args.split}.npz")
    x = data["x"]
    y = data["y_engagement"]

    logits = predict_torch(args.checkpoint, x)
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

    fp32_macro = f1_score(y, y_pred, average="macro", zero_division=0)
    print(f"{args.split}: fp32 macro-F1 {fp32_macro:.4f}")
    print(json.dumps({"auc": aucs}, indent=1))

    if args.int8 is not None:
        int8_logits = predict_onnx(args.int8, x)
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
