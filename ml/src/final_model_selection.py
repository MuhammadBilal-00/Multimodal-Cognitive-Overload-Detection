"""Phase 5 of the rigorous-fix pass: final model selection via pairwise
clip-level bootstrap significance testing (reuses significance.py's
clip_bootstrap/macro_f1/ci95 utilities directly, not rewritten) between
the three candidates Phases 3-4 produced: the Optuna-tuned TCN
(final_candidate.py, seed=42 checkpoint), gradient boosting (the
strongest classical baseline throughout this project), and the stacked
ensemble (ensemble_stack.py) -- on the official Validation split.

Usage: python -u ml/src/final_model_selection.py [--iterations 2000]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
REPO_ROOT = SRC_DIR.parent.parent

from baselines import aggregate_features  # noqa: E402
from ensemble_stack import find_phase3_seed42_run  # noqa: E402
from model import build_model  # noqa: E402
from significance import ci95, clip_bootstrap, macro_f1  # noqa: E402

DATASET_DIR = REPO_ROOT / "artifacts" / "dataset"
RESULTS_DIR = REPO_ROOT / "docs" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    best = json.loads((RESULTS_DIR / "cv_hyperparameter_search_best.json").read_text())
    params = best["params"]

    train = np.load(DATASET_DIR / "Train.npz")
    val = np.load(DATASET_DIR / "Validation.npz")
    x_train, y_train = train["x"], train["y_engagement"]
    x_val, y_val, clip_ids_val = (val["x"], val["y_engagement"], val["clip_ids"])

    print("loading tuned TCN (Phase 3 seed=42)...")
    tcn_run = find_phase3_seed42_run(params)
    tcn = build_model("tcn", n_features=13, channels=params["channels"],
                      dropout=params["dropout"])
    tcn.load_state_dict(torch.load(tcn_run / "best.pt", map_location="cpu",
                                   weights_only=True))
    tcn.eval()
    with torch.no_grad():
        logits, _ = tcn(torch.from_numpy(x_val))
        tcn_val_probs = torch.softmax(logits, dim=1).numpy()
    tcn_pred = tcn_val_probs.argmax(1)

    print("training gradient boosting on full Train...")
    x_train_agg, x_val_agg = aggregate_features(x_train), aggregate_features(x_val)
    gbm = HistGradientBoostingClassifier(random_state=42)
    gbm.fit(x_train_agg, y_train,
            sample_weight=compute_sample_weight("balanced", y_train))
    gbm_pred = gbm.predict(x_val_agg)

    print("building ensemble prediction (reusing cached OOF + meta-learner logic)...")
    oof = np.load(REPO_ROOT / "artifacts" / "runs" / "ensemble_oof_cache.npz")["oof"]
    meta = LogisticRegression(max_iter=2000)
    meta.fit(oof, y_train)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    rf.fit(x_train_agg, y_train)
    ensemble_val_features = np.concatenate(
        [tcn_val_probs, rf.predict_proba(x_val_agg), gbm.predict_proba(x_val_agg)],
        axis=1)
    ensemble_pred = meta.predict(ensemble_val_features)

    candidates = {"tuned_tcn": tcn_pred, "gradient_boosting": gbm_pred,
                 "stacked_ensemble": ensemble_pred}
    points = {name: macro_f1(y_val, pred) for name, pred in candidates.items()}
    print(f"point estimates: {points}")

    rng = np.random.default_rng(args.seed)
    pairs = [("tuned_tcn", "gradient_boosting"),
             ("tuned_tcn", "stacked_ensemble"),
             ("gradient_boosting", "stacked_ensemble")]
    results = {"points": {k: round(v, 4) for k, v in points.items()},
              "iterations": args.iterations, "pairwise": {}}
    for a, b in pairs:
        f1s_a, f1s_b = clip_bootstrap(y_val, candidates[a], candidates[b],
                                      clip_ids_val, args.iterations, rng)
        diff = f1s_a - f1s_b
        p = min(float(2 * min((diff <= 0).mean(), (diff >= 0).mean())), 1.0)
        results["pairwise"][f"{a}_vs_{b}"] = {
            f"{a}_ci_95": ci95(f1s_a), f"{b}_ci_95": ci95(f1s_b),
            "diff_point": round(points[a] - points[b], 4),
            "diff_ci_95": ci95(diff), "two_sided_p_value": round(p, 4),
        }
        print(f"{a} vs {b}: diff={points[a]-points[b]:+.4f} p={p:.4f}")

    winner = max(points, key=points.get)
    results["winner_by_point_estimate"] = winner
    results["method"] = ("paired cluster bootstrap, resampling unit = clip ID, "
                         "reusing significance.py's clip_bootstrap()")

    out_path = RESULTS_DIR / "final_model_selection.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {out_path}")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
