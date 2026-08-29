"""Phase 3 of the rigorous-fix pass: retrain Phase 2's Optuna-selected
hyperparameter configuration at FULL production budget (train.py's own
100-epoch/patience-15 defaults, not the search's reduced 25/6 budget)
across multiple seeds, on the official Train split, evaluated on the
official Validation split. This is the actual final candidate -- Phase 2
found the configuration on a fast, reduced-fold/reduced-epoch proxy;
train.py (already verified, unchanged) trains the real thing.

Reads docs/results/cv_hyperparameter_search_best.json (Phase 2's winner)
and shells out to train.py once per seed, reusing the exact,
already-verified training loop. Resumable the same way
multi_seed_robustness.py is: an existing run dir matching this exact
config+seed is reused rather than retrained.

Usage: python ml/src/final_candidate.py [--seeds 42 7 123 99 17]
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
RESULTS_DIR = REPO_ROOT / "docs" / "results"
BEST_JSON = RESULTS_DIR / "cv_hyperparameter_search_best.json"


def find_existing_run(params: dict, seed: int) -> Path | None:
    for run_dir in sorted(RUNS_DIR.glob("2*")):
        config_path, report_path = run_dir / "config.json", run_dir / "report.json"
        if not config_path.exists() or not report_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue
        if (config.get("seed") == seed
                and config.get("arch", "tcn") == "tcn"
                and config.get("feature_subset", "full") == "full"
                and not config.get("ordinal", False)
                and abs(config.get("channels", 64) - params["channels"]) < 1e-9
                and abs(config.get("dropout", 0.2) - params["dropout"]) < 1e-6
                and abs(config.get("lr", 1e-3) - params["lr"]) < 1e-9
                and abs(config.get("weight_power", 1.0) - params["weight_power"]) < 1e-6
                and abs(config.get("state_loss_weight", 0.5)
                       - params["state_loss_weight"]) < 1e-6
                and abs(config.get("label_smoothing", 0.0)
                       - params["label_smoothing"]) < 1e-6
                and config.get("batch_size", 128) == params["batch_size"]):
            return run_dir
    return None


def train(params: dict, seed: int) -> Path:
    existing = find_existing_run(params, seed)
    if existing is not None:
        print(f"reusing existing run for seed={seed}: {existing}")
        return existing
    cmd = [sys.executable, str(SRC_DIR / "train.py"),
           "--seed", str(seed),
           "--channels", str(params["channels"]),
           "--dropout", str(params["dropout"]),
           "--lr", str(params["lr"]),
           "--weight-power", str(params["weight_power"]),
           "--state-loss-weight", str(params["state_loss_weight"]),
           "--label-smoothing", str(params["label_smoothing"]),
           "--batch-size", str(params["batch_size"])]
    print(f"training seed={seed}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("run dir: "):
            return Path(line[len("run dir: "):].strip())
    raise RuntimeError(f"no 'run dir:' line for seed={seed}\n"
                       f"{result.stdout[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 7, 123, 99, 17])
    args = parser.parse_args()

    best = json.loads(BEST_JSON.read_text())
    params = best["params"]
    print(f"Phase 2 winning config (CV macro_f1={best['value']:.4f}): {params}")

    rows = [["seed", "macro_f1", "accuracy", "qwk"]]
    for seed in args.seeds:
        run_dir = train(params, seed)
        report = json.loads((run_dir / "report.json").read_text())
        val = report["val"]
        rows.append([seed, round(val["macro_f1"], 4), round(val["accuracy"], 4),
                    round(val["qwk"], 4)])
        print(f"  seed={seed} macro_f1={val['macro_f1']:.4f}")

    f1s = [r[1] for r in rows[1:]]
    mean, std = statistics.mean(f1s), statistics.stdev(f1s)
    rows.append(["mean", round(mean, 4), "", ""])
    rows.append(["std", round(std, 4), "", ""])

    out_path = RESULTS_DIR / "final_candidate.csv"
    with open(out_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"wrote {out_path}")
    print(f"final candidate: mean={mean:.4f} std={std:.4f} (n={len(f1s)} seeds)")
    print(f"vs. shipped TCN (multi_seed_robustness.csv): mean=0.3015 std=0.0049")


if __name__ == "__main__":
    main()
