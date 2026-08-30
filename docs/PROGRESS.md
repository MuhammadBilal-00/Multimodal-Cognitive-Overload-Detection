# Project Progress — as of 2026-08-30

Privacy-preserving cognitive state detection in the browser: a temporal
convolutional network trained on DAiSEE, quantized to int8 ONNX, running
entirely client-side on MediaPipe-extracted facial features — video never
leaves the user's machine.

- **Track A** — Bilal: `ml/` — DAiSEE video → 13-feature extraction →
  TCN → quantized `model_int8.onnx` + `scaler.json`
- **Track B** — Azeem: `web/` — Next.js 14 + TypeScript, in-browser
  feature extraction, inference, dashboard, benchmarking

`CONTRACT.md` is the frozen interface contract between the tracks
(currently **v1.4** — Amendments 1–4; Amendment 3 fixed the states channel
order, Amendment 4 the brow eye-centre formula. Both partners signed
2026-08-03; Azeem's sign-off on Amendments 3 and 4 is still open).

## Timeline

| Day(s) | Date | What happened |
|---|---|---|
| 1 | 2026-08-01 | Repo scaffold, interface contract, pinned Python 3.11 env (`ml/requirements.txt`). DAiSEE access form submitted, download started. Audio check: DAiSEE is video-only, so "multimodal" = three visual modality families (CONTRACT.md §8). |
| 2 | 2026-08-01 | `features.py` (13-feature contract implementation) + 17 unit tests. Landmark indices visually verified — `docs/verification/landmarks_{overview,zoom}.png`. |
| 3–5 | 2026-08-01→02 | Multiprocess extraction over 9,032 clips: **0 failures, 99.96% face detection** (`docs/results/extraction_stats.json`). Windowed dataset + `scaler.json` (pitch_centre 0.3733). Parity fixture (VP9 WebM) for the Python↔browser gate. |
| 6–7 | 2026-08-02 | TCN model, 41.5k params. A6.5 browser smoke PASS — static QDQ int8 (dynamic quantization is browser-incompatible: onnxruntime-web WASM lacks ConvInteger). J1 parity gate PASS, worst feature diff **0.0079** at the time (tol 0.02; later 0.0157 after the numFaces fix, and **0.0035** after Amendment 4 — the current committed value) — `docs/results/parity_report.json`; GPU-delegate variant fails (0.05), which is why the app pins the CPU delegate. |
| 8–9 | 2026-08-02 | Training: 6 runs. Winner = weighted CE (full inverse-freq), lr 1e-3 — val macro-F1 **0.3061** vs majority floor 0.1813. Focal loss, lower lr, sqrt weights, label smoothing all worse. |
| 10–11 | 2026-08-02 | Validation eval artifacts (`docs/results/metrics_validation.csv`, confusion + ROC plots). Trained int8 shipped to `web/public/model/` (60 KB, quantization Δ macro-F1 −0.0015). Next.js app + fake-webcam e2e PASS. |
| 14–15 | 2026-08-02 | **Final test-set evaluation — run exactly once, after model freeze**: see headline table below. J3 browser benchmarks archived (`docs/results/browser_benchmark.json`). |
| — | 2026-08-01→03 | In parallel, Track B built the full web app in its own repository against a dummy model with contract-identical I/O (`scripts/make_dummy_onnx.py`), so it was never blocked on training. |
| — | 2026-08-03 | **Repository merge** (see below), CONTRACT.md v1.1 amendment, multi-face support, 3 s prediction cadence. |
| — | 2026-08-09 | **Gap-closure pass** (see below): J1 rebuilt for real (caught and fixed a `numFaces` regression), A5 baselines, J3 refreshed, J2 fixed, CI added. CONTRACT.md v1.2 (Amendment 2). |

## Headline results (Test split, 14,241 windows, evaluated once)

| Metric | Value |
|---|---|
| Macro-F1 (fp32) | 0.2475 |
| Macro-F1 (int8, shipped) | 0.2460 |
| Majority-class floor | 0.1655 |
| 3-class merged macro-F1 | 0.3318 |
| Model size fp32 → int8 | 163 KB → 60 KB |
| In-browser inference (p50) | 0.475 ms |
| Feature parity Python↔browser | max diff 0.0035 (tol 0.02) |

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
  (`model_fp32.onnx` is produced into gitignored `artifacts/export/` for
  the Python-side fp32-vs-int8 latency comparison, not shipped to the browser)

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
- `web/harness/*.html` and `model_fp32.onnx` carried over untouched (historical:
  `web/harness/onnx_smoke.html` was later deleted and then restored in
  2026-08-30's audit pass, because `ml/scripts/browser_tests.py` still needs it).

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

## Gap-closure — 2026-08-09

A self-audit against live repo state (not just recollection) found that
`cc4f5c8`'s merge had left three browser-driving scripts stale — pointing
at routes/globals the merge itself deleted — and that the recorded J1
"PASS" therefore validated code that no longer existed. Full writeup:
`GAP_CLOSURE_PLAN.md`. Fixed:

- **J1 rebuilt** as a real Playwright Test (`web/tests/e2e/features.parity.test.ts`,
  `web/app/parity-test/`) against 100 pre-extracted static PNG frames
  instead of a seeked `<video>` (headless Chromium doesn't reliably
  present seeked frames). Reuses the production landmarker factory rather
  than a hand-rolled second instance — and that reuse **caught a real,
  previously-undetected regression**: production's `numFaces: 4`
  (multi-face overlay, commit `c16cd9a`) shifts landmarks enough to fail
  parity on blink frames (`gaze_y` diff up to 0.86 vs 0.016 at
  `numFaces: 1`). Fixed by splitting `lib/faceLandmarker.ts` into
  `createFeatureLandmarker()` (`numFaces: 1`, feeds the model — matches
  `ml/src/extract.py`) and `createDisplayLandmarker()` (`numFaces: 4`,
  overlay/People-count only); `usePipeline.ts` now runs both. Tolerance
  (0.02) and this split are recorded as CONTRACT.md Amendment 2.
- **A5 baselines** (`ml/src/baselines.py`, new) — logistic regression +
  random forest on 65-dim aggregate features, both Validation and Test
  splits, `docs/results/baselines.csv`. Majority-class rows cross-checked
  exactly against `metrics_{validation,test}.csv`'s own majority baseline
  (0.1813 / 0.1655). The TCN (val macro-F1 0.3061) beats both classical
  baselines (logreg 0.242, RF 0.2669).
- **J3 refreshed** — `web/lib/benchmark.ts` now records
  `hardwareConcurrency`/`deviceMemory`; `ml/scripts/collect_benchmark.py`
  rewritten against the current app (was targeting a deleted `/bench`
  page) and driven by `page.on("download")`, not a `window` global. One
  real run recorded: `docs/benchmarks/benchmark-dev-i7-13700H-16GB.json`.
  Runbook for the other two required machines: `docs/benchmarks/README.md`.
- **J2 fixed** — `ml/scripts/e2e_app_test.py` was polling a
  `window.__ENGINE_STATE` that didn't exist and clicking a "Start camera"
  button the app no longer has (camera now auto-starts). Added a minimal
  `window.__ENGINE_STATE` mirror in `usePipeline.ts`
  (`{status, prediction, facePresent}`), retargeted the script, fixed a
  leftover 2 Hz-era `time.sleep(2)` to 4 s (must exceed the current 3 s
  inference window). Re-ran: fresh `app_e2e.json` / `artifacts/app_screenshot_e2e.png`,
  `ok: true`.
- **CI added** (`.github/workflows/ci.yml`) — vitest + typecheck and
  `pytest ml/tests/` run unconditionally; the J1 Playwright gate runs only
  when `ml/tests/fixtures/parity_frames/*.png` are present on the runner,
  since those PNGs are DAiSEE-derived and (like the clip they're sampled
  from) can never be committed to git under the dataset license — a
  constraint that predates this session and was never actually solved by
  the original design either (there was no CI at all before). Skips with
  a visible warning rather than silently passing or failing every push.
- Fixed a stale doc claim: `docs/architecture.md` said "GPU delegate with
  CPU fallback"; the app has been CPU-only since the numFaces work in
  `c16cd9a`, and CPU-only is required for J1 parity regardless (Amendment 2).

## Phase 1 hardening — 2026-08-09 (PROJECT_COMPLETION_PLAN.md)

All three Phase 1 sub-phases done same day as gap-closure. Full findings:
`docs/demo-failure-modes.md`.

- **1.1 Clean-machine install** — ran the README verbatim on a genuine
  fresh `git clone` (not this session's working copy). Found the
  documented torch-CPU workaround needs to be the primary install
  command, not a fallback (a plain `pip install -r ml/requirements.txt`
  fails outright on a clean machine); fixed in README.md. Verified
  `npm install && npm run build && npm start` end to end, including a
  real fake-webcam prediction against the freshly built app.
- **1.2 Cross-browser** — Chrome, Edge, Firefox all load, run, and honor
  COOP/COEP (all report `wasm×20`, confirming multithreaded WASM
  everywhere). Chrome/Edge got real face detection via a file-backed
  fake camera; Firefox has no equivalent in Playwright, so its "no face"
  path was confirmed but real-face detection there needs one manual
  check before the defense. Safari skipped (no Mac).
- **1.3 Failure modes** — no face, two faces (synthetic composite — the
  one natural DAiSEE candidate never has its background person facing
  the camera), bad lighting (moderate and severe), and glasses all
  tested with real DAiSEE clips driven through a clean build. None
  crashed, froze, or produced a non-normalized/garbage prediction.

## UI redesign and states-channel fix — 2026-08-10 → 2026-08-16

Previously missing from this record (its absence was itself an audit
finding — Appendix F cites this file as the full day-by-day history):

- **2026-08-10** (`e9f6ba7`): full web UI redesign — light Meet-style
  theme, real trend chart, hideable stats panels; 9 components rewritten.
  (A same-day commit `2be9ffb` carries the message "asd" — a sloppy
  commit message on a minor follow-up, acknowledged rather than hidden.)
- **2026-08-16** (`2e95fe4`): the states-channel-order defect found and
  fixed — `PredictionPanel` listed the four states alphabetically while
  the model emits Boredom/Engagement/Confusion/Frustration, so the
  "Confused" bar had been displaying P(engagement). Fixed with
  `web/lib/states.ts` as the single mapping site, guarded by a test that
  parses `ml/src/labels.py`, documented as CONTRACT.md Amendment 3; the
  states head was also retrained with per-channel pos_weight (its rare
  channels had collapsed to base rates), Validation artefacts refreshed.

## Extended model comparison, rigorous search, honest evaluation — 2026-08-26 → 2026-08-29

Three commits (`dd2c99b`, `9294347`, `1c17a19`) plus this session's
evaluation-reframing pass, all Track A / `ml/` + `docs/results/` only —
the shipped browser model was deliberately **not** changed (every search
result validated it rather than beating it):

- **Extended comparison** (`docs/results/model_comparison_summary.md`):
  gradient-boosting baseline (significantly beats the TCN on Test,
  p<0.001), QWK metric, LSTM/GRU/Transformer at matched budgets,
  feature-family ablation, 3-seed robustness reruns, clip-level bootstrap
  significance testing.
- **Rigorous search** (`docs/results/rigorous_model_search.md`):
  subject-grouped 5-fold CV (caught + fixed a clip-vs-subject fold-leakage
  bug that had manufactured a false feature-selection finding), 40-trial
  Optuna search (validated the shipped hyperparameters: tuned-vs-shipped
  p=0.92), OOF stacking ensemble (failed; failure diagnosed to the
  meta-learner over-trusting RF's majority-class confidence; remedy
  recovers it only to parity), CORAL ordinal variant (trade-off, not a
  win). Optuna added to `ml/requirements.txt`.
- **Honest evaluation reframing** (`ml/src/clip_eval.py`,
  `docs/results/clip_eval_{validation,test}.json`): clip-level scoring
  (the benchmark's actual granularity) + per-class decision thresholds
  calibrated on Validation, applied frozen to the consumed-once Test
  predictions: **Test macro-F1 0.2475 → 0.2829, accuracy 36.9% → 44.7%**,
  deployed model untouched. Binary screening reported with AUC/balanced
  accuracy (accuracy alone is below the trivial baseline at this
  prevalence — stated, not hidden).
- **Thesis updated in place** (`docs/thesis/FYP_Report.md`): Experiments
  7–8 added, §4.3.3 claim corrected, §5 rewritten around the new
  evidence, CORAL/Optuna references added, AI-use declaration added
  (wording to be verified against university policy).

## Live rehearsal fix — 2026-08-29

First live rehearsal (real webcam, production build) surfaced the one
imperfect behaviour the 2026-08-09 hardening pass had flagged as an
observation: stepping out of frame showed the model's arbitrary output
for an all-empty window ("Frustrated 99%") beneath the no-face notice.
Fixed: `PredictionPanel` now suppresses readings entirely on
`ood.noFace` (em-dash placeholders + explanatory notice) and the trend
chart skips no-face points. Thesis Figure 4.4 recaptured accordingly.

## Artifact index

| Artifact | What it is |
|---|---|
| `docs/results/extraction_stats.json` | 9,032-clip extraction: failures, detection rate |
| `docs/results/class_dist.png` | DAiSEE label distribution |
| `docs/results/parity_report.json` | J1 Python↔browser feature parity (rebuilt harness, `numFaces:1` feature path, post-Amendment-4 brow fix, PASS — worst diff 0.0035) |
| `docs/results/parity_report_gpu.json` | Same gate with GPU delegate (FAIL — why CPU is pinned) |
| `docs/results/baselines.csv` | A5: majority / logreg / random-forest macro-F1 + accuracy, Validation & Test |
| `docs/results/quantization.csv` | fp32 → int8 accuracy delta |
| `docs/results/metrics_{validation,test}.csv` | Per-class P/R/F1, macro-F1, AUCs |
| `docs/results/confusion_{validation,test}.png` | Confusion matrices |
| `docs/results/roc_{validation,test}.png` | ROC curves |
| `docs/results/browser_smoke.json` | A6.5: int8 ONNX runs in onnxruntime-web |
| `docs/results/browser_benchmark.json` | Legacy J3 artifact (deleted `/bench` page era: int8 vs fp32 + separate landmark timing) — superseded by `docs/benchmarks/`, kept for history |
| `docs/results/app_e2e.json` | J2 fake-webcam end-to-end app test (refreshed 2026-08-09). Its screenshot stays in gitignored `artifacts/` because the fake-cam frame is a DAiSEE participant's face; the committed, licence-safe capture of the current UI is `docs/results/app_screenshot_synthetic.png` |
| `docs/verification/landmarks_{overview,zoom}.png` | Visual landmark-index verification |
| `docs/benchmarks/benchmark-dev-i7-13700H-16GB.json` | J3: real int8 model, this dev machine (2026-08-09) |
| `docs/benchmarks/benchmark-dummy-model-dev-machine.json` | Legacy Track B benchmark harness output (dummy-model era) — kept for history, not a valid J3 data point |
| `docs/benchmarks/README.md` | J3 runbook: how to add the other two required machines |

## Open items

- **Two more J3 machines** — BUILD_PLAN_1.md §J3 wants ≥3 machines; only
  this dev machine is recorded. Runbook: `docs/benchmarks/README.md`.
  Blocked on machine availability (PROJECT_COMPLETION_PLAN.md Phase 2).
- **Thesis writeup — first full draft done (2026-08-09)**, once the
  university report template and marking scheme were supplied:
  `docs/thesis/FYP_Report.md` / `.docx`, 14,352 words main body
  (Chapters 1–5; target 10,000–15,000) / 16,400 total including front
  matter, references and appendices, Harvard referencing throughout (13
  citations, each individually verified real via web search before use —
  none fabricated), mapped onto the template's exact chapter structure.
  Front matter completed 2026-08-30 (Acknowledgements, generated
  contents lists, page numbers, document properties); all figures are
  embedded. Remaining: the supervisor review pass, the university's
  required declaration-of-originality and ethics wording, and the
  authorship-consistency question — see `SUBMISSION_CHECKLIST.md` §1b.
- **One manual Firefox check** — Playwright can't feed Firefox a
  file-backed fake camera the way it can Chromium, so Firefox's
  real-face detection (as opposed to load/init/no-face-path) hasn't been
  independently confirmed — see `docs/demo-failure-modes.md`.
- **Student handoff session** — materials ready
  (`docs/student-handoff.md`: pipeline walkthrough + question bank +
  `baselines.py` deep-dive); the actual session with the student hasn't
  happened yet.
- **`v1.0` tag** — the live dry-run rehearsal it was gated behind is
  **done** (2026-08-29, all failure modes passed —
  `docs/dry-run-checklist.md`). The tag itself is the only remaining
  Phase 5 item, and is deliberately held until the human-only items in
  `SUBMISSION_CHECKLIST.md` are settled.
