"""Real hyperparameter search with Optuna (Phase 2 of the rigorous-fix
pass): TPE-sampled Bayesian search + median pruning, evaluated on Phase
0's subject-level 5-fold CV (not a single split), on the feature preset
Phase 1 selected.

Phase 1's result (docs/results/cv_feature_selection_summary.csv, paired
significance tests in the accompanying log): no feature preset beat any
other significantly under proper subject-level CV (all pairwise p >
0.11) -- the earlier single-split/3-seed "geometric+gaze beats full"
finding did not survive. This search therefore runs on the FULL 13
features (the current shipped default), the only defensible choice absent
a significant feature-selection result.

Pruning is two-level: Optuna prunes a trial mid-fold (epoch-level, via
cv_train.py's trial.report()/should_prune() on every epoch) AND between
folds (fold-level, checked after each fold completes) -- together these
keep total compute far below "every trial runs to completion on every
fold."

Resilient to interruption: every completed trial's result is persisted to
the Optuna storage (a local SQLite file) as it happens, and re-running
this script resumes the existing study rather than starting over --
Optuna's own `load_if_exists=True` mechanism, not custom code (unlike
cv_feature_selection.py, which needed hand-rolled resume logic because it
doesn't use a tool with built-in persistence).

Usage: python -u ml/src/cv_hyperparameter_search.py [--n-trials 40]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from cv_splits import N_FOLDS, get_folds  # noqa: E402
from cv_train import train_and_evaluate  # noqa: E402
from feature_groups import PRESETS  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
STORAGE = f"sqlite:///{REPO_ROOT / 'artifacts' / 'runs' / 'optuna_cv_search.db'}"
STUDY_NAME = "tcn_cv_hyperparameter_search"
FEATURE_PRESET = "full"  # Phase 1's result: no preset won significantly


def objective(trial: optuna.Trial, x: np.ndarray, y: np.ndarray, ys: np.ndarray,
             clip_ids: np.ndarray, n_features: int, max_epochs: int,
             patience: int, search_folds: int) -> float:
    hp = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "channels": trial.suggest_int("channels", 32, 128, step=16),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "weight_power": trial.suggest_float("weight_power", 0.3, 1.0),
        "state_loss_weight": trial.suggest_float("state_loss_weight", 0.1, 1.0),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
    }

    fold_f1s = []
    # Only the first `search_folds` of the 5 CV folds are used here, purely
    # for search speed -- this environment interrupts long background jobs
    # unpredictably, and a full 5-fold trial was not surviving long enough
    # to ever finish (see docs/results/rigorous_model_search.md for the
    # observed timings). Phase 1 already used the full 5 folds for feature
    # selection, and Phase 3 retrains the winning config at full budget on
    # the official Train/Validation split -- this reduced-fold objective is
    # a search proxy, not a final evaluation.
    for fold_idx, (train_idx, val_idx) in enumerate(get_folds(clip_ids, y)):
        if fold_idx >= search_folds:
            break
        metrics = train_and_evaluate(
            x[train_idx], y[train_idx], ys[train_idx],
            x[val_idx], y[val_idx], ys[val_idx],
            hp=hp, n_features=n_features, max_epochs=max_epochs,
            patience=patience, trial=trial,
            report_offset=fold_idx * max_epochs)
        fold_f1s.append(metrics["macro_f1"])
        # Fold-level pruning: after each fold, let Optuna judge the
        # running mean against other trials at this same fold count.
        trial.report(float(np.mean(fold_f1s)), step=10_000 + fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_f1s))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=25,
                        help="per-fold epoch cap (lower for a quick smoke test)")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--search-folds", type=int, default=2,
                        help="how many of the 5 CV folds each trial "
                             "actually trains on (search speed trade-off; "
                             "see objective()'s docstring comment)")
    args = parser.parse_args()

    data = np.load(DATASET_DIR / "Train.npz")
    feature_idx = PRESETS[FEATURE_PRESET]
    x = data["x"][:, :, feature_idx]
    y, ys, clip_ids = data["y_engagement"], data["y_states"], data["clip_ids"]

    (REPO_ROOT / "artifacts" / "runs").mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE, direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        load_if_exists=True)

    # Count only finished trials toward the target -- a trial left RUNNING
    # by an interrupted process (this environment interrupts long
    # background jobs periodically) never completes on its own and would
    # otherwise silently eat one of the n_trials slots on every resume.
    finished = sum(1 for t in study.trials
                  if t.state in (optuna.trial.TrialState.COMPLETE,
                                optuna.trial.TrialState.PRUNED))
    remaining = args.n_trials - finished
    print(f"{finished} finished trials ({len(study.trials)} total, "
          f"{len(study.trials) - finished} orphaned/running); "
          f"running {max(remaining, 0)} more (target {args.n_trials})")
    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, x, y, ys, clip_ids,
                                    len(feature_idx), args.max_epochs,
                                    args.patience, args.search_folds),
            n_trials=remaining)

    out_dir = REPO_ROOT / "docs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [["trial", "state", "value"] + list(study.trials[0].params.keys())]
    for t in study.trials:
        rows.append([t.number, t.state.name,
                    round(t.value, 4) if t.value is not None else ""]
                    + [t.params.get(k, "") for k in rows[0][3:]])
    with open(out_dir / "cv_hyperparameter_search.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    best = study.best_trial
    with open(out_dir / "cv_hyperparameter_search_best.json", "w") as fh:
        json.dump({"value": best.value, "params": best.params,
                  "feature_preset": FEATURE_PRESET,
                  "n_features": len(feature_idx)}, fh, indent=1)

    print(f"wrote {out_dir / 'cv_hyperparameter_search.csv'} and _best.json")
    print(f"best trial #{best.number}: macro_f1={best.value:.4f}")
    print(json.dumps(best.params, indent=1))


if __name__ == "__main__":
    main()
