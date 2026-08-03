# Project Progress — as of 2026-08-03

Privacy-preserving cognitive state detection in the browser: a temporal
convolutional network trained on DAiSEE, quantized to int8 ONNX, running
entirely client-side on MediaPipe-extracted facial features — video never
leaves the user's machine.

- **Track A** — Bilal: `ml/` — DAiSEE video → 13-feature extraction →
  TCN → quantized `model_int8.onnx` + `scaler.json`
- **Track B** — Azeem: `web/` — Next.js 14 + TypeScript, in-browser
  feature extraction, inference, dashboard, benchmarking

`CONTRACT.md` is the frozen interface contract between the tracks
(currently **v1.1**, both partners signed 2026-08-03).

## Timeline

| Day(s) | Date | What happened |
|---|---|---|
| 1 | 2026-08-01 | Repo scaffold, interface contract, pinned Python 3.11 env (`ml/requirements.txt`). DAiSEE access form submitted, download started. Audio check: DAiSEE is video-only, so "multimodal" = three visual modality families (CONTRACT.md §8). |
| 2 | 2026-08-01 | `features.py` (13-feature contract implementation) + 17 unit tests. Landmark indices visually verified — `docs/verification/landmarks_{overview,zoom}.png`. |
| 3–5 | 2026-08-01→02 | Multiprocess extraction over 9,032 clips: **0 failures, 99.96% face detection** (`docs/results/extraction_stats.json`). Windowed dataset + `scaler.json` (pitch_centre 0.3733). Parity fixture (VP9 WebM) for the Python↔browser gate. |
| 6–7 | 2026-08-02 | TCN model, 41.5k params. A6.5 browser smoke PASS — static QDQ int8 (dynamic quantization is browser-incompatible: onnxruntime-web WASM lacks ConvInteger). J1 parity gate PASS, worst feature diff **0.0079** (tol 0.02) — `docs/results/parity_report.json`; GPU-delegate variant fails (0.05), which is why the app pins the CPU delegate. |
| 8–9 | 2026-08-02 | Training: 6 runs. Winner = weighted CE (full inverse-freq), lr 1e-3 — val macro-F1 **0.3061** vs majority floor 0.1813. Focal loss, lower lr, sqrt weights, label smoothing all worse. |
| 10–11 | 2026-08-02 | Validation eval artifacts (`docs/results/metrics_validation.csv`, confusion + ROC plots). Trained int8 shipped to `web/public/model/` (60 KB, quantization Δ macro-F1 −0.0016). Next.js app + fake-webcam e2e PASS. |
| 14–15 | 2026-08-02 | **Final test-set evaluation — run exactly once, after model freeze**: see headline table below. J3 browser benchmarks archived (`docs/results/browser_benchmark.json`). |
| — | 2026-08-01→03 | In parallel, Track B built the full web app in its own repository against a dummy model with contract-identical I/O (`scripts/make_dummy_onnx.py`), so it was never blocked on training. |
| — | 2026-08-03 | **Repository merge** (see below), CONTRACT.md v1.1 amendment, multi-face support, 3 s prediction cadence. |

## Headline results (Test split, 14,241 windows, evaluated once)

| Metric | Value |
|---|---|
| Macro-F1 (fp32) | 0.2475 |
| Macro-F1 (int8, shipped) | 0.2460 |
| Majority-class floor | 0.1655 |
| 3-class merged macro-F1 | 0.3318 |
| Model size fp32 → int8 | 163 KB → 60 KB |
| In-browser inference (p50) | 0.4 ms |
| Feature parity Python↔browser | max diff 0.0079 (tol 0.02) |

Caveat for the writeup: engagement class 0 has only 4 test clips, so its
per-class metrics are unmeasurable; adjacent-class confusion dominates the
errors (`docs/results/confusion_test.png`).

## Track A deliverables

- `ml/src/` — `features.py` (reference implementation of CONTRACT §2–4),
  `extract.py` (multiprocess, resumable), `labels.py`, `dataset.py`,
  `model.py` (TCN), `train.py`, `eval.py`, `export.py` (ONNX,
  dynamo=False; static QDQ int8 quantization)
- `ml/tests/` — unit tests + the parity fixture
- `ml/scripts/` — landmark verification, parity-fixture builder, headless
  browser gates, e2e app test, benchmark collector
- Shipped: `web/public/model/model_int8.onnx` (60 KB) + `scaler.json`
  (+ `model_fp32.onnx` for browser comparison)

## Track B deliverables

- Full pipeline (all in-browser): webcam → MediaPipe FaceLandmarker
  (478 landmarks, WASM) → `features.ts` (13 floats, ported line-for-line
  from CONTRACT §2–4) → 30-frame ring buffer → standardise → onnxruntime-web
  → softmax/sigmoid in JS → dashboard. Stage-by-stage breakdown in
  `docs/architecture.md`.
- Privacy hardening: CSP `connect-src 'self'` + COOP/COEP; blocks a real,
  undisclosed MediaPipe telemetry call — evidence in `docs/privacy.md`.
- Dashboard: prediction panel with 60 s engagement sparkline, live feature
  panel, performance HUD, benchmark harness (300 cycles, p50/p95/p99, heap
  delta, JSON export — `docs/benchmarks/`).
- Unit tests (vitest): features, ring buffer, math utils, scaler
  validation, primary-face selection.

## Repository merge — 2026-08-03, commit `cc4f5c8`

The two tracks had developed in **unrelated git histories** (Azeem's Track B
lived in its own GitHub repo, not a fork). Merged with
`--allow-unrelated-histories`:

- `CONTRACT.md` / `README.md`: took Track B's version (pure superset —
  Azeem's sign-off + Track B docs; nothing removed).
- `web/` became Azeem's TypeScript app; Bilal's old JS smoke-test scaffold
  (`web/app/page.js`, `layout.js`, `bench/page.js`, `web/src/*.js`) was
  deleted as superseded.
- The **real trained** `model_int8.onnx` + `scaler.json` were kept over
  Track B's dummy placeholder.
- `web/harness/*.html` and `model_fp32.onnx` carried over untouched.

## Post-merge changes — 2026-08-03

- **3 s prediction cadence** (`2b27662`): inference stride 5 → 30 samples —
  one prediction per non-overlapping 3.0 s window instead of 2 Hz.
  Recorded as CONTRACT.md §6 Amendment 1 (v1.0 → v1.1); training stride
  and `ml/` untouched.
- **Multi-face support** (`c16cd9a`): landmarker `numFaces` 1 → 4; overlays
  on every detected face (primary cyan, others dimmed); "People" count in
  the perf HUD. Prediction remains single, on the primary face — largest
  bbox with centroid stickiness + size hysteresis (`web/lib/primaryFace.ts`).
- **CPU-only landmarker** (same commit): removed the GPU-first delegate;
  the model was trained on CPU/XNNPACK-extracted features and the GPU
  delegate fails the J1 parity gate (`docs/results/parity_report_gpu.json`).

## Artifact index

| Artifact | What it is |
|---|---|
| `docs/results/extraction_stats.json` | 9,032-clip extraction: failures, detection rate |
| `docs/results/class_dist.png` | DAiSEE label distribution |
| `docs/results/parity_report.json` | J1 Python↔browser feature parity (CPU delegate, PASS) |
| `docs/results/parity_report_gpu.json` | Same gate with GPU delegate (FAIL — why CPU is pinned) |
| `docs/results/quantization.csv` | fp32 → int8 accuracy delta |
| `docs/results/metrics_{validation,test}.csv` | Per-class P/R/F1, macro-F1, AUCs |
| `docs/results/confusion_{validation,test}.png` | Confusion matrices |
| `docs/results/roc_{validation,test}.png` | ROC curves |
| `docs/results/browser_smoke.json` | A6.5: int8 ONNX runs in onnxruntime-web |
| `docs/results/browser_benchmark.json` | J3: in-browser latency percentiles |
| `docs/results/app_e2e.json`, `app_screenshot.png` | Fake-webcam end-to-end app test |
| `docs/verification/landmarks_{overview,zoom}.png` | Visual landmark-index verification |
| `docs/benchmarks/benchmark-dummy-model-dev-machine.json` | Track B benchmark harness output (dummy model era) |

## Open items

- **A5 classical baselines** — reserved for the FYP student; inputs ready
  in `artifacts/dataset/`.
- **Thesis writeup** — methodology, error analysis (class 0 sparsity,
  adjacent-class confusion), results tables from `docs/results/`.
- **Stale browser-gate scripts** — `ml/scripts/e2e_app_test.py`,
  `browser_tests.py`, `collect_benchmark.py` still target the deleted JS
  scaffold routes and must be rewritten against the current app; when they
  are, prediction-wait timeouts must account for the 3 s cadence.
- Re-run J3 benchmarks / e2e against the merged app with the real model on
  the current UI.
