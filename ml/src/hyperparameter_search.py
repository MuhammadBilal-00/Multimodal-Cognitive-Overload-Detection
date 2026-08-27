"""Lightweight hyperparameter search (Phase B3): does the shipped TCN
config (channels=64, dropout=0.2) and the gradient-boosting baseline's
plain sklearn defaults actually sit near a local optimum, or would modest
tuning move either number meaningfully? No new dependency -- sklearn's
`RandomizedSearchCV` (already pinned) for gradient boosting, and a small
manual grid (3 new configs, the 64/0.2 baseline reused from the existing
architecture_comparison.csv row rather than retrained) for the TCN via
`train.py --channels/--dropout`, reusing the exact, already-verified
training loop rather than a second copy of it.

Usage: python ml/src/hyperparameter_search.py
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from baselines import aggregate_features  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
LABELS = [0, 1, 2, 3]

GBM_PARAM_DIST = {
    "learning_rate": [0.01, 0.03, 0.1, 0.2, 0.3],
    "max_iter": [50, 100, 200, 300],
    "max_depth": [None, 3, 5, 8],
    "l2_regularization": [0.0, 0.1, 1.0],
}

# The (64, 0.2) baseline is NOT retrained here -- it's the exact config
# already trained for architecture_comparison.csv, reused for this table.
TCN_GRID = [
    {"channels": 32, "dropout": 0.2},   # narrower
    {"channels": 96, "dropout": 0.2},   # wider
    {"channels": 64, "dropout": 0.4},   # more dropout
]


def load_split(split: str) -> tuple:
    data = np.load(DATASET_DIR / f"{split}.npz")
    return data["x"], data["y_engagement"]


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                          zero_division=0))


def gbm_search() -> list:
    x_train, y_train = load_split("Train")
    x_val, y_val = load_split("Validation")
    x_train_agg = aggregate_features(x_train)
    x_val_agg = aggregate_features(x_val)
    sample_weight = compute_sample_weight("balanced", y_train)

    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=42), GBM_PARAM_DIST,
        n_iter=20, scoring="f1_macro", cv=3, random_state=42, n_jobs=-1)
    search.fit(x_train_agg, y_train, sample_weight=sample_weight)

    default = HistGradientBoostingClassifier(random_state=42)
    default.fit(x_train_agg, y_train, sample_weight=sample_weight)

    rows = [["model", "params", "macro_f1", "accuracy", "qwk"]]
    for name, estimator, params in (
            ("gbm_default", default, "sklearn defaults"),
            ("gbm_randomsearch_best", search.best_estimator_,
             json.dumps(search.best_params_))):
        pred = estimator.predict(x_val_agg)
        rows.append([name, params, round(macro_f1(y_val, pred), 4),
                    round(float((y_val == pred).mean()), 4),
                    round(float(cohen_kappa_score(y_val, pred,
                                                  weights="quadratic")), 4)])
    return rows


def run_tcn_config(config: dict) -> Path:
    cmd = [sys.executable, str(SRC_DIR / "train.py")]
    for k, v in config.items():
        cmd += [f"--{k}", str(v)]
    print(f"training TCN grid config {config}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("run dir: "):
            return Path(line[len("run dir: "):].strip())
    raise RuntimeError(f"no 'run dir:' line for config {config}\n"
                       f"{result.stdout[-2000:]}")


def tcn_grid() -> list:
    rows = [["channels", "dropout", "parameters", "macro_f1", "accuracy", "qwk"]]

    with open(RESULTS_DIR / "architecture_comparison.csv", newline="") as fh:
        baseline = next(r for r in csv.DictReader(fh) if r["arch"] == "tcn")
    rows.append([64, 0.2, baseline["parameters"], baseline["macro_f1"],
                baseline["accuracy"], baseline["qwk"]])

    for config in TCN_GRID:
        run_dir = run_tcn_config(config)
        report = json.loads((run_dir / "report.json").read_text())
        cfg = json.loads((run_dir / "config.json").read_text())
        val = report["val"]
        rows.append([config["channels"], config["dropout"], cfg["parameters"],
                    round(val["macro_f1"], 4), round(val["accuracy"], 4),
                    round(val["qwk"], 4)])
    return rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gbm_rows = gbm_search()
    with open(RESULTS_DIR / "hyperparameter_search.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(gbm_rows)
    print(f"wrote {RESULTS_DIR / 'hyperparameter_search.csv'}")
    for r in gbm_rows:
        print(r)

    tcn_rows = tcn_grid()
    with open(RESULTS_DIR / "tcn_grid.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(tcn_rows)
    print(f"wrote {RESULTS_DIR / 'tcn_grid.csv'}")
    for r in tcn_rows:
        print(r)


if __name__ == "__main__":
    main()
