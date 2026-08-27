"""Feature-family ablation (extends A5/A7): does each of CONTRACT.md
§2/§8's three claimed visual modality families (geometric, pose, gaze)
actually earn its place, or is "multimodal" here an architectural framing
without measured support?

Four configs (feature_groups.PRESETS): geometric-only, geometric+pose,
geometric+gaze, full 13. Each is run through (a) the classical baselines
(fast, in-process, matching baselines.py's models/hyperparameters) and (b)
the TCN at its winning hyperparameters (weighted CE, lr 1e-3, seed 42 --
train.py's own defaults). The "full" TCN row reuses the project's existing
winning run (artifacts/runs/20260801_185630) rather than retraining it;
the other three configs are trained fresh by shelling out to
`train.py --feature-subset`, reusing the exact, already-verified training
loop rather than a second copy of it.

Validation split only -- this is model-development-adjacent work, so it
follows the same "Test touched once" discipline baselines.py's own
--split default does.

Usage: python ml/src/feature_ablation.py
"""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from baselines import aggregate_features  # noqa: E402
from feature_groups import PRESETS  # noqa: E402
from model import build_model  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
LABELS = [0, 1, 2, 3]
# The project's existing winning TCN run (train.py's defaults, "full"
# feature set) -- reused here rather than retrained.
FULL_TCN_RUN_DIR = RUNS_DIR / "20260801_185630"


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                          zero_division=0))


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def load_split(split: str, feature_idx: list) -> tuple:
    data = np.load(DATASET_DIR / f"{split}.npz")
    return data["x"][:, :, feature_idx], data["y_engagement"]


def run_classical(preset: str, rows: list) -> None:
    feature_idx = PRESETS[preset]
    x_train, y_train = load_split("Train", feature_idx)
    x_val, y_val = load_split("Validation", feature_idx)
    x_train_agg = aggregate_features(x_train)
    x_val_agg = aggregate_features(x_val)

    models = {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=2000),
        "random_forest": RandomForestClassifier(class_weight="balanced",
                                                random_state=42),
    }
    for name, clf in models.items():
        clf.fit(x_train_agg, y_train)
        pred = clf.predict(x_val_agg)
        rows.append([preset, name, len(feature_idx),
                    round(macro_f1(y_val, pred), 4),
                    round(float(accuracy_score(y_val, pred)), 4),
                    round(qwk(y_val, pred), 4)])

    gbm = HistGradientBoostingClassifier(random_state=42)
    gbm.fit(x_train_agg, y_train,
            sample_weight=compute_sample_weight("balanced", y_train))
    pred = gbm.predict(x_val_agg)
    rows.append([preset, "gradient_boosting", len(feature_idx),
                round(macro_f1(y_val, pred), 4),
                round(float(accuracy_score(y_val, pred)), 4),
                round(qwk(y_val, pred), 4)])


def predict_tcn(checkpoint: Path, n_features: int, x: np.ndarray) -> np.ndarray:
    model = build_model("tcn", n_features=n_features)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                     weights_only=True))
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            logits, _ = model(torch.from_numpy(x[i:i + 512]))
            preds.append(logits.argmax(dim=1).numpy())
    return np.concatenate(preds)


def run_tcn_fresh(preset: str) -> Path:
    cmd = [sys.executable, str(SRC_DIR / "train.py"),
           "--feature-subset", preset]
    print(f"training TCN on preset={preset}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("run dir: "):
            return Path(line[len("run dir: "):].strip())
    raise RuntimeError(f"no 'run dir:' line in train.py output for "
                       f"preset={preset}\n{result.stdout[-2000:]}")


def run_tcn(preset: str, rows: list) -> None:
    feature_idx = PRESETS[preset]
    x_val, y_val = load_split("Validation", feature_idx)
    run_dir = FULL_TCN_RUN_DIR if preset == "full" else run_tcn_fresh(preset)
    pred = predict_tcn(run_dir / "best.pt", len(feature_idx), x_val)
    rows.append([preset, "tcn", len(feature_idx),
                round(macro_f1(y_val, pred), 4),
                round(float(accuracy_score(y_val, pred)), 4),
                round(qwk(y_val, pred), 4)])


def main() -> None:
    rows = [["feature_subset", "model", "n_features", "macro_f1",
             "accuracy", "qwk"]]
    for preset in PRESETS:
        run_classical(preset, rows)
        run_tcn(preset, rows)

    out_path = REPO_ROOT / "docs" / "results" / "feature_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    print(f"wrote {out_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
