"""Multi-seed robustness check (Phase A of the model-comparison extension):
every architecture/ablation number produced so far is a SINGLE training run
per config (seed 42 only). This re-runs the configs a conclusion is being
drawn from at 2 additional seeds (7, 123) and reports mean/std/min/max
alongside the existing seed-42 point estimate -- specifically to check
whether the geometric+gaze TCN result beating the full 13-feature TCN
(0.3252 vs 0.3061, docs/results/feature_ablation.csv) survives across seeds
or was one run's variance.

Two groups:
  - architecture: tcn/lstm/gru/transformer on the full feature set, seeds 7
    and 123 (8 new runs), combined with the existing seed-42 point estimates
    already in architecture_comparison.csv (must include a "transformer"
    row before this script runs -- see ml/src/train.py --arch transformer).
  - feature_ablation: TCN ONLY (the architecture with the surprising
    non-monotonic result -- LSTM/GRU/Transformer robustness is a smaller,
    secondary question, handled by the architecture group above), all 4
    presets, seeds 7 and 123 (8 new runs), combined with the existing
    seed-42 point estimates already in feature_ablation.csv.

Drives new runs via the same subprocess-and-parse-stdout pattern
architecture_comparison.py/feature_ablation.py already use -- reuses the
exact, already-verified training loop rather than a second copy of it.

Usage: python ml/src/multi_seed_robustness.py
"""

import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "results"

NEW_SEEDS = [7, 123]
EXISTING_SEED = 42
ARCHS = ["tcn", "lstm", "gru", "transformer"]
PRESETS = ["geometric", "geometric_pose", "geometric_gaze", "full"]


RUNS_DIR = REPO_ROOT / "artifacts" / "runs"


def find_existing_run(arch: str, feature_subset: str, seed: int) -> Path | None:
    """An already-completed run dir matching this exact config, if one
    exists -- avoids re-training after an interrupted previous attempt, and
    lets the architecture group's tcn-on-"full" run at a given seed double
    as the feature-ablation group's "full"-preset run at that same seed
    (they are the identical config), rather than training it twice.
    """
    for run_dir in sorted(RUNS_DIR.glob("2*")):
        config_path, report_path = run_dir / "config.json", run_dir / "report.json"
        if not config_path.exists() or not report_path.exists():
            continue  # incomplete/interrupted run
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue
        if (config.get("arch", "tcn") == arch
                and config.get("feature_subset", "full") == feature_subset
                and config.get("seed") == seed
                and config.get("channels", 64) == 64
                and config.get("dropout", 0.2) == 0.2
                and not config.get("ordinal", False)):
            return run_dir
    return None


def run_train(extra_args: list) -> Path:
    cmd = [sys.executable, str(SRC_DIR / "train.py")] + extra_args
    print(f"training: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("run dir: "):
            return Path(line[len("run dir: "):].strip())
    raise RuntimeError(f"no 'run dir:' line for {extra_args}\n{result.stdout[-2000:]}")


def get_or_train(arch: str, feature_subset: str, seed: int) -> Path:
    existing = find_existing_run(arch, feature_subset, seed)
    if existing is not None:
        print(f"reusing existing run for arch={arch} feature_subset="
              f"{feature_subset} seed={seed}: {existing}")
        return existing
    extra = ["--seed", str(seed)]
    if arch != "tcn":
        extra += ["--arch", arch]
    if feature_subset != "full":
        extra += ["--feature-subset", feature_subset]
    return run_train(extra)


def metrics_from_report(run_dir: Path) -> dict:
    report = json.loads((run_dir / "report.json").read_text())
    val = report["val"]
    return {"macro_f1": val["macro_f1"], "accuracy": val["accuracy"],
            "qwk": val["qwk"]}


def read_csv_by_key(path: Path, key_col: str) -> dict:
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows[row[key_col]] = {"macro_f1": float(row["macro_f1"]),
                                  "accuracy": float(row["accuracy"]),
                                  "qwk": float(row["qwk"])}
    return rows


def main() -> None:
    long_rows = [["experiment", "config", "seed", "macro_f1", "accuracy", "qwk"]]

    existing_arch = read_csv_by_key(RESULTS_DIR / "architecture_comparison.csv", "arch")
    missing = [a for a in ARCHS if a not in existing_arch]
    if missing:
        raise SystemExit(
            f"architecture_comparison.csv is missing rows for {missing} -- "
            f"train them first (e.g. `python ml/src/train.py --arch transformer`) "
            f"and add the row before running this script.")

    for arch in ARCHS:
        m = existing_arch[arch]
        long_rows.append(["architecture", arch, EXISTING_SEED,
                          m["macro_f1"], m["accuracy"], m["qwk"]])
        for seed in NEW_SEEDS:
            run_dir = get_or_train(arch, "full", seed)
            m2 = metrics_from_report(run_dir)
            long_rows.append(["architecture", arch, seed,
                              m2["macro_f1"], m2["accuracy"], m2["qwk"]])

    existing_ablation = {
        row["feature_subset"]: {"macro_f1": float(row["macro_f1"]),
                                "accuracy": float(row["accuracy"]),
                                "qwk": float(row["qwk"])}
        for row in csv.DictReader(open(RESULTS_DIR / "feature_ablation.csv", newline=""))
        if row["model"] == "tcn"
    }
    for preset in PRESETS:
        m = existing_ablation[preset]
        long_rows.append(["feature_ablation", preset, EXISTING_SEED,
                          m["macro_f1"], m["accuracy"], m["qwk"]])
        for seed in NEW_SEEDS:
            run_dir = get_or_train("tcn", preset, seed)
            m2 = metrics_from_report(run_dir)
            long_rows.append(["feature_ablation", preset, seed,
                              m2["macro_f1"], m2["accuracy"], m2["qwk"]])

    out_path = RESULTS_DIR / "multi_seed_robustness.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        csv.writer(fh).writerows(long_rows)
    print(f"wrote {out_path}")

    groups = defaultdict(list)
    for exp, config, seed, f1, acc, qwk in long_rows[1:]:
        groups[(exp, config)].append(f1)
    print("\nsummary (macro_f1 mean +/- std [min, max], n=3 seeds):")
    for (exp, config), values in groups.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {exp:16s} {config:16s} {mean:.4f} +/- {std:.4f}  "
              f"[{min(values):.4f}, {max(values):.4f}]")


if __name__ == "__main__":
    main()
