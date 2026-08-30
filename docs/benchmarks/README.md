# Benchmarks — J3 (BUILD_PLAN_1.md §J3 / §B7)

"≥30 FPS" is meaningless without stating the hardware it was measured on.
Every file in this directory is one machine's run of the app's own
in-browser benchmark (`BenchmarkPanel.tsx` → `lib/benchmark.ts` → "Run 300
inferences": 300 ONNX-Runtime-Web inference cycles against the shipped
`model_int8.onnx`, reporting mean/p50/p95/p99 ms, mean FPS, and JS heap
delta), collected either by hand (click the button, save the download) or
headlessly via `ml/scripts/collect_benchmark.py`.

## Required: at least 3 machines

BUILD_PLAN_1.md §J3 calls for both laptops plus one lab/library PC. As of
2026-08-09 only this dev machine has a result
(`benchmark-dev-i7-13700H-16GB.json`). The other two are still needed —
run the collector below on each and commit the resulting JSON. Do not
fabricate numbers for hardware that hasn't actually run this.

## Running it

**Automated (recommended — headless, reproducible):**

```powershell
# 1. Build the app once (the collector only starts the production server,
#    it does not build):
cd web
npm install
npm run build

# 2. From the repo root, with the Python venv active (Playwright is in
#    ml/requirements.txt):
cd ..
python ml\scripts\collect_benchmark.py --machine-label "laptop2-ryzen5-8GB"
```

This starts `next start`, opens the app in headless Chromium, clicks "Run
300 inferences", captures the JSON it downloads, and saves it to
`docs/benchmarks/benchmark-<machine-label>.json`. `--machine-label` is
free text — use it to identify the machine (CPU model, RAM); it becomes
part of the filename (non-alphanumeric characters replaced with `-`).

**Manual (if Playwright isn't set up on that machine):**

```powershell
cd web
npm install
npm run build
npm start
```

Open `http://localhost:3000`, click the **Benchmark** toggle in the control
bar to expand the panel, then click "Run 300 inferences" and save the JSON
it downloads into this directory as `benchmark-<machine-label>.json`.
(The panel starts collapsed — the automated collector had to be fixed for
the same reason on 2026-08-30.)

## What's in each file

```jsonc
{
  "cycles": 300,
  "meanMs": 0.57, "p50": 0.47, "p95": 1.16, "p99": 2.12,
  "meanFps": 1762.3,
  "heapDeltaMB": 0,
  "backend": "wasm", "threads": 20,
  "userAgent": "...",
  "timestamp": "2026-08-09T16:33:35.157Z",
  "hardwareConcurrency": 20,   // navigator.hardwareConcurrency — logical cores
  "deviceMemory": 16           // navigator.deviceMemory (GB) — Chromium-only, rough bucket
}
```

`hardwareConcurrency` / `deviceMemory` are read from inside the page
(`navigator.*`), so they're only available in the JSON itself, not
something a driver script can inject after the fact — but they're rough
proxies (`deviceMemory` in particular is a bucketed, capped value some
browsers don't expose at all), not exact hardware identification. Still
record the actual CPU model and RAM size in `--machine-label` /
the commit message.

## Superseded artifact

`benchmark-dummy-model-dev-machine.json` was collected against the
placeholder dummy ONNX model (`scripts/make_dummy_onnx.py`) before the
real trained model shipped (2026-08-03 merge). Kept for history; not a
valid J3 data point — use the `dev-i7-13700H-16GB` file (or later,
correctly-labeled files) for the thesis's hardware comparison table.
