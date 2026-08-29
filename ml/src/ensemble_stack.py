"""Phase 4 of the rigorous-fix pass: out-of-fold (OOF) stacking ensemble
-- the most direct attempt at actually fixing "the core deliverable is
weak" (issue #1), not just documenting it.

For each of Phase 0's 5 subject-level CV folds: train the tuned TCN
(Phase 2/3's winning hyperparameters, docs/results/cv_hyperparameter_
search_best.json), gradient boosting, and random forest on the other 4
folds; predict class PROBABILITIES on the held-out fold. Collecting this
across all 5 folds covers every Train window with a probability vector
from a model that never saw it during its own training -- zero leakage,
by construction (each window's OOF prediction comes only from models
fit on the other folds).

A multinomial LogisticRegression meta-learner is then trained on the
concatenated OOF probability vectors (4 classes x 3 base learners = 12
features per window) against the true labels -- this is what "stacking"
means: learning how to weight/combine the base learners' probability
outputs, rather than a fixed average.

Final evaluation: retrain all three base learners on the FULL official
Train split (reusing Phase 3's seed=42 tuned-TCN checkpoint for the TCN
rather than retraining a 4th time), get their Validation-split
probability predictions, feed through the already-fitted meta-learner,
score against Validation labels -- directly comparable to every other
Validation-split number in this project's results.

Usage: python -u ml/src/ensemble_stack.py
"""

import csv
import json
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
from cv_splits import N_FOLDS as N_FOLDS_CONST  # noqa: E402
from cv_splits import get_folds  # noqa: E402
from cv_train import DEFAULT_HP, train_and_evaluate  # noqa: E402
from model import build_model  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
OOF_CACHE = RUNS_DIR / "ensemble_oof_cache.npz"
N_CLASSES = 4
LABELS = list(range(N_CLASSES))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=LABELS,
                          zero_division=0))


def qwk(y_true, y_pred) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def tcn_train_predict_proba(x_train, y_train, ys_train, x_val, hp: dict,
                            n_features: int, max_epochs: int = 30,
                            patience: int = 6, seed: int = 42) -> np.ndarray:
    """Trains a TCN on (x_train, y_train, ys_train) via cv_train's loop
    machinery (reused, not duplicated) but returns Validation-set softmax
    PROBABILITIES rather than train_and_evaluate()'s scored metrics dict --
    stacking needs probabilities, not just argmax predictions.

    max_epochs/patience match cv_hyperparameter_search.py's search budget
    (not final_candidate.py's full 100-epoch production budget): the
    128-channel tuned config costs ~45-55s/epoch on a ~80%-of-Train fold,
    so 60/10 (this function's first version) risked well over an hour per
    fold against an environment that interrupts long background jobs
    periodically -- observed directly: fold 0 alone took ~1h at that
    budget. This is a probability-generator for the meta-learner, not the
    final model, and the search budget already demonstrated adequate
    signal (0.30-0.34 macro-F1 range across 40 trials at this budget).
    """
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from train import inverse_frequency_weights

    torch.manual_seed(seed)
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train),
                             torch.from_numpy(ys_train))
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)

    class_weights = inverse_frequency_weights(train_ds.tensors[1],
                                              power=hp["weight_power"])
    states_pos = train_ds.tensors[2].sum(dim=0)
    states_pos_weight = (len(train_ds) - states_pos) / states_pos.clamp(min=1)

    model = build_model("tcn", n_features=n_features, channels=hp["channels"],
                        dropout=hp["dropout"])
    engagement_loss = nn.CrossEntropyLoss(weight=class_weights,
                                          label_smoothing=hp["label_smoothing"])
    states_loss = nn.BCEWithLogitsLoss(pos_weight=states_pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    for epoch in range(max_epochs):
        model.train()
        for x, y_eng, y_states in train_loader:
            optimizer.zero_grad()
            logits_eng, logits_states = model(x)
            loss = (engagement_loss(logits_eng, y_eng)
                    + hp["state_loss_weight"] * states_loss(logits_states, y_states))
            loss.backward()
            if hp["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(x_val))
        probs = torch.softmax(logits, dim=1).numpy()
    return probs


def build_oof(x, y, ys, clip_ids, hp: dict) -> np.ndarray:
    """(N, 12) OOF probability matrix: [tcn_p0..3, rf_p0..3, gbm_p0..3].

    Resumable per-fold (not just all-or-nothing): each completed fold's
    slice is written to OOF_CACHE immediately, along with a bitmask of
    which folds are done, so an interruption mid-run (this environment
    interrupts long background jobs periodically -- five TCN trainings in
    one script is exactly the shape that's been hit before) loses at most
    one partial fold, not all prior progress.
    """
    n = len(y)
    if OOF_CACHE.exists():
        cached = np.load(OOF_CACHE)
        oof, folds_done = cached["oof"], cached["folds_done"]
        print(f"resuming OOF cache: folds done so far = "
              f"{folds_done.tolist()}")
    else:
        oof = np.zeros((n, 3 * N_CLASSES), dtype=np.float32)
        folds_done = np.zeros(N_FOLDS_CONST, dtype=bool)

    for fold_idx, (train_idx, val_idx) in enumerate(get_folds(clip_ids, y)):
        if folds_done[fold_idx]:
            print(f"OOF fold {fold_idx} already done, skipping")
            continue
        print(f"OOF fold {fold_idx}: training base learners...", flush=True)
        x_agg_train = aggregate_features(x[train_idx])
        x_agg_val = aggregate_features(x[val_idx])
        sw = compute_sample_weight("balanced", y[train_idx])

        rf = RandomForestClassifier(class_weight="balanced", random_state=42)
        rf.fit(x_agg_train, y[train_idx])
        gbm = HistGradientBoostingClassifier(random_state=42)
        gbm.fit(x_agg_train, y[train_idx], sample_weight=sw)

        tcn_probs = tcn_train_predict_proba(
            x[train_idx], y[train_idx], ys[train_idx], x[val_idx],
            hp=hp, n_features=x.shape[-1])

        oof[val_idx, 0:4] = tcn_probs
        oof[val_idx, 4:8] = rf.predict_proba(x_agg_val)
        oof[val_idx, 8:12] = gbm.predict_proba(x_agg_val)
        folds_done[fold_idx] = True

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OOF_CACHE, oof=oof, folds_done=folds_done)
        print(f"OOF fold {fold_idx} done, cached", flush=True)

    return oof


def full_train_base_learners(x, y, ys, hp: dict) -> dict:
    """Retrains all 3 base learners on the FULL Train split for final
    evaluation. TCN reuses Phase 3's seed=42 checkpoint rather than
    retraining a 4th time (identical config, already trained at full
    production budget in final_candidate.py)."""
    x_agg = aggregate_features(x)
    sw = compute_sample_weight("balanced", y)
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf.fit(x_agg, y)
    gbm = HistGradientBoostingClassifier(random_state=42)
    gbm.fit(x_agg, y, sample_weight=sw)

    tcn_run_dir = find_phase3_seed42_run(hp)
    tcn_model = build_model("tcn", n_features=x.shape[-1],
                            channels=hp["channels"], dropout=hp["dropout"])
    tcn_model.load_state_dict(torch.load(tcn_run_dir / "best.pt",
                                         map_location="cpu", weights_only=True))
    tcn_model.eval()
    return {"rf": rf, "gbm": gbm, "tcn": tcn_model}


def find_phase3_seed42_run(hp: dict) -> Path:
    for run_dir in sorted(RUNS_DIR.glob("2*")):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        if (config.get("seed") == 42 and config.get("arch", "tcn") == "tcn"
                and abs(config.get("channels", 64) - hp["channels"]) < 1e-9
                and abs(config.get("lr", 1e-3) - hp["lr"]) < 1e-9):
            return run_dir
    raise RuntimeError("Phase 3's seed=42 tuned-TCN run not found -- "
                       "run final_candidate.py first")


def main() -> None:
    best = json.loads((RESULTS_DIR / "cv_hyperparameter_search_best.json").read_text())
    hp = DEFAULT_HP | best["params"]

    train_data = np.load(DATASET_DIR / "Train.npz")
    x, y, ys, clip_ids = (train_data["x"], train_data["y_engagement"],
                          train_data["y_states"], train_data["clip_ids"])

    print("building OOF predictions (5-fold)...")
    oof = build_oof(x, y, ys, clip_ids, hp)

    print("training meta-learner on OOF predictions...")
    # sklearn >=1.7 always fits multinomial logistic regression for a
    # multi-class problem with a real solver (no more explicit
    # multi_class= flag / no more one-vs-rest fallback) -- this is the
    # multinomial meta-learner the design calls for by default.
    #
    # DELIBERATELY UNWEIGHTED (class_weight=None), unlike every other
    # classifier in this project: all three base learners already train
    # with their own class-imbalance correction (RF/GBM's
    # class_weight="balanced", the TCN's inverse-frequency-weighted loss),
    # so their probability outputs are already imbalance-adjusted. Stacking
    # a SECOND class_weight="balanced" meta-learner on top double-corrects
    # -- verified directly: OOF macro_f1 0.3134 (unweighted) vs. 0.2368
    # (balanced) on the identical cached OOF matrix, with the balanced
    # version's coefficients ~3x larger in magnitude (up to 9.19), i.e.
    # genuinely destabilised, not just a smaller effect.
    meta = LogisticRegression(max_iter=2000)
    meta.fit(oof, y)

    oof_pred = meta.predict(oof)
    print(f"OOF (in-sample-of-stack, out-of-fold-of-base) macro_f1="
         f"{macro_f1(y, oof_pred):.4f}")

    print("retraining base learners on full Train, evaluating on Validation...")
    val_data = np.load(DATASET_DIR / "Validation.npz")
    x_val, y_val = val_data["x"], val_data["y_engagement"]
    learners = full_train_base_learners(x, y, ys, hp)

    x_val_agg = aggregate_features(x_val)
    rf_probs = learners["rf"].predict_proba(x_val_agg)
    gbm_probs = learners["gbm"].predict_proba(x_val_agg)
    with torch.no_grad():
        logits, _ = learners["tcn"](torch.from_numpy(x_val))
        tcn_probs = torch.softmax(logits, dim=1).numpy()

    val_features = np.concatenate([tcn_probs, rf_probs, gbm_probs], axis=1)
    ensemble_pred = meta.predict(val_features)

    rows = [["model", "macro_f1", "accuracy", "qwk"]]
    rows.append(["tcn_tuned", round(macro_f1(y_val, tcn_probs.argmax(1)), 4),
                round(float(accuracy_score(y_val, tcn_probs.argmax(1))), 4),
                round(qwk(y_val, tcn_probs.argmax(1)), 4)])
    rows.append(["random_forest", round(macro_f1(y_val, rf_probs.argmax(1)), 4),
                round(float(accuracy_score(y_val, rf_probs.argmax(1))), 4),
                round(qwk(y_val, rf_probs.argmax(1)), 4)])
    rows.append(["gradient_boosting", round(macro_f1(y_val, gbm_probs.argmax(1)), 4),
                round(float(accuracy_score(y_val, gbm_probs.argmax(1))), 4),
                round(qwk(y_val, gbm_probs.argmax(1)), 4)])
    rows.append(["stacked_ensemble", round(macro_f1(y_val, ensemble_pred), 4),
                round(float(accuracy_score(y_val, ensemble_pred)), 4),
                round(qwk(y_val, ensemble_pred), 4)])

    out_path = RESULTS_DIR / "ensemble_comparison.csv"
    with open(out_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"wrote {out_path}")
    for r in rows:
        print(r)

    meta_path = RESULTS_DIR / "ensemble_meta_learner_coef.json"
    with open(meta_path, "w") as fh:
        json.dump({
            "classes": meta.classes_.tolist(),
            "coef_shape": list(meta.coef_.shape),
            "feature_order": ["tcn_p0", "tcn_p1", "tcn_p2", "tcn_p3",
                             "rf_p0", "rf_p1", "rf_p2", "rf_p3",
                             "gbm_p0", "gbm_p1", "gbm_p2", "gbm_p3"],
            "coef": meta.coef_.tolist(),
        }, fh, indent=1)
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
