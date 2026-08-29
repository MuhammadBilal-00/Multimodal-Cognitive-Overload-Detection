"""fp32-vs-int8 latency comparison (Track B brief §10: "fp32 vs int8
benchmark comparison" — quantization.csv covers size and F1 only; this
covers speed). Python onnxruntime, CPU EP, same 300 real standardised
validation windows through both graphs, wall-clock per single-window run.

Not the browser number (docs/benchmarks/*.json is that); this isolates the
quantization effect itself on identical hardware/runtime, which the
browser benchmark can't do (only the int8 model ships).

Usage: python ml/scripts/latency_fp32_vs_int8.py
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPORT_DIR = REPO_ROOT / "artifacts" / "export"
OUT_PATH = REPO_ROOT / "docs" / "benchmarks" / "fp32_vs_int8_latency.json"
N_RUNS = 300
WARMUP = 20


def bench(model_path: Path, windows: np.ndarray) -> dict:
    session = ort.InferenceSession(str(model_path),
                                   providers=["CPUExecutionProvider"])
    for i in range(WARMUP):
        session.run(None, {"features": windows[i % len(windows)]})
    times = []
    for i in range(N_RUNS):
        x = windows[i % len(windows)]
        t0 = time.perf_counter()
        session.run(None, {"features": x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "model": model_path.name,
        "size_bytes": model_path.stat().st_size,
        "runs": N_RUNS,
        "mean_ms": round(statistics.mean(times), 4),
        "p50_ms": round(times[len(times) // 2], 4),
        "p95_ms": round(times[int(len(times) * 0.95)], 4),
        "p99_ms": round(times[int(len(times) * 0.99)], 4),
    }


def main() -> None:
    data = np.load(REPO_ROOT / "artifacts" / "dataset" / "Validation.npz")["x"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data), size=min(N_RUNS, len(data)), replace=False)
    windows = [data[i:i + 1].astype(np.float32) for i in idx]

    results = {
        "note": ("Python onnxruntime CPU EP, single-window inference, "
                 "300 real standardised Validation windows, 20-run warmup. "
                 "Isolates the quantization latency effect on identical "
                 "hardware/runtime; browser-side numbers live in the other "
                 "files in this directory."),
        "onnxruntime_version": ort.__version__,
        "fp32": bench(EXPORT_DIR / "model_fp32.onnx", windows),
        "int8": bench(EXPORT_DIR / "model_int8.onnx", windows),
    }
    results["speedup_p50"] = round(
        results["fp32"]["p50_ms"] / results["int8"]["p50_ms"], 3)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {OUT_PATH}")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
