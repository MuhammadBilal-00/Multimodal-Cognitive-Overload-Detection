"""Ordinal-regression (CORAL) comparison (Phase B2): DAiSEE's engagement
label is ordinal (very low < low < engaged < very engaged); QWK (already
added to eval.py/baselines.py/train.py) MEASURES ordinal-aware performance
but the shipped TCN still TARGETS the label as 4-way nominal classification
(softmax + cross-entropy). This checks whether directly optimising for the
ordinal structure -- CORAL (Cao, Mirjalili and Raschka, 2020), implemented
in ml/src/model.py's CoralLayer/coral_loss/coral_predict, wired up via
`train.py --ordinal` -- changes predictive performance.

Trains the TCN once with --ordinal at the standard hyperparameters (seed
42, full feature set) and reports it alongside the already-committed
standard (nominal-softmax) TCN result in architecture_comparison.csv, on
the same metrics.

Usage: python ml/src/ordinal_comparison.py
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent
RESULTS_DIR = REPO_ROOT / "docs" / "results"


def run_ordinal() -> Path:
    cmd = [sys.executable, str(SRC_DIR / "train.py"), "--ordinal"]
    print(f"training ordinal TCN: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=True,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("run dir: "):
            return Path(line[len("run dir: "):].strip())
    raise RuntimeError(f"no 'run dir:' line for ordinal run\n{result.stdout[-2000:]}")


def main() -> None:
    rows = [["model", "macro_f1", "accuracy", "qwk"]]

    with open(RESULTS_DIR / "architecture_comparison.csv", newline="") as fh:
        existing = next(r for r in csv.DictReader(fh) if r["arch"] == "tcn")
    rows.append(["tcn_softmax (nominal, existing)", existing["macro_f1"],
                existing["accuracy"], existing["qwk"]])

    run_dir = run_ordinal()
    report = json.loads((run_dir / "report.json").read_text())
    val = report["val"]
    rows.append(["tcn_coral (ordinal, new)", round(val["macro_f1"], 4),
                round(val["accuracy"], 4), round(val["qwk"], 4)])

    out_path = RESULTS_DIR / "ordinal_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    print(f"wrote {out_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
