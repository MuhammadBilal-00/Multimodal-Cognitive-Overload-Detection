"""Architecture-family comparison (LSTM/GRU vs. the shipped TCN): does the
TCN's dilated-convolution structure actually predict engagement better than
a standard recurrent baseline of comparable width, or would a same-width
LSTM/GRU do just as well? The closest prior DAiSEE work (Abedi and Khan,
2021, already cited in the thesis) compares a ResNet+TCN hybrid against a
ResNet+LSTM hybrid on raw video and finds the TCN wins by 2.75 points; this
checks whether the same TCN-over-recurrent advantage holds on this
project's own 13-feature geometric representation, rather than only citing
that it held on someone else's pixel-based one.

Reuses the project's existing winning TCN run
(artifacts/runs/20260801_185630 -- train.py's own defaults: weighted CE,
lr 1e-3, seed 42, state_loss_weight 0.5) rather than retraining it; the
later states-retrained checkpoint used a non-default state_loss_weight and
is intentionally NOT used here, so the "tcn" row is trained under the
identical regime the fresh LSTM/GRU runs below use. LSTM and GRU
(EngagementRNN, ml/src/model.py) are trained fresh once each, at the same
defaults, by shelling out to `train.py --arch`, reusing the exact,
already-verified training loop.

Validation split only, matching baselines.py's own default.

Usage: python ml/src/architecture_comparison.py
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from model import build_model  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
LABELS = [0, 1, 2, 3]
TCN_RUN_DIR = RUNS_DIR / "20260801_185630"


def predict(arch: str, checkpoint: Path, n_features: int,
           x: np.ndarray) -> np.ndarray:
    model = build_model(arch, n_features=n_features)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            logits, _ = model(torch.from_numpy(x[i:i + 512]))
            preds.append(logits.argmax(dim=1).numpy())
    return np.concatenate(preds)


def metrics_for(arch: str, checkpoint: Path, n_features: int) -> dict:
    data = np.load(DATASET_DIR / "Validation.npz")
    x, y = data["x"], data["y_engagement"]
    pred = predict(arch, checkpoint, n_features, x)
    return {
        "macro_f1": round(float(f1_score(y, pred, average="macro",
                                         labels=LABELS, zero_division=0)), 4),
        "accuracy": round(float((y == pred).mean()), 4),
        "qwk": round(float(cohen_kappa_score(y, pred, weights="quadratic")), 4),
    }


def run_arch(arch: str) -> Path:
    cmd = [sys.executable, str(SRC_DIR / "train.py"), "--arch", arch]
    print(f"training arch={arch}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("run dir: "):
            return Path(line[len("run dir: "):].strip())
    raise RuntimeError(f"no 'run dir:' line in train.py output for "
                       f"arch={arch}\n{result.stdout[-2000:]}")


def main() -> None:
    rows = [["arch", "parameters", "macro_f1", "accuracy", "qwk"]]

    tcn_config = json.loads((TCN_RUN_DIR / "config.json").read_text())
    n_features = tcn_config.get("n_features", 13)  # predates the field; was always 13
    tcn_metrics = metrics_for("tcn", TCN_RUN_DIR / "best.pt", n_features)
    rows.append(["tcn", tcn_config["parameters"], tcn_metrics["macro_f1"],
                tcn_metrics["accuracy"], tcn_metrics["qwk"]])

    for arch in ("lstm", "gru"):
        run_dir = run_arch(arch)
        config = json.loads((run_dir / "config.json").read_text())
        metrics = metrics_for(arch, run_dir / "best.pt", config["n_features"])
        rows.append([arch, config["parameters"], metrics["macro_f1"],
                    metrics["accuracy"], metrics["qwk"]])

    out_path = REPO_ROOT / "docs" / "results" / "architecture_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    print(f"wrote {out_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
