# Gap-closure plan — J1 parity, A5 baselines, J3 benchmarks, J2 e2e, CI

> **Superseded 2026-08-09 by commit `9a5e635`** ("test: rebuild J1/J2/J3
> gates, add A5 baselines, wire up CI") — everything below is done. Kept as
> a historical record of the reasoning, not as an open task list.

Companion to `BUILD_PLAN_1.md`. Written 2026-08-09, revised same day after a self-audit against live repo state. This file records what's broken, why, and the fix plan — not yet implemented.

## Context

The status review found:

1. **J1 (feature parity gate)** — the committed harness (`web/harness/parity.html`) imports a file deleted in the `cc4f5c8` merge, so the "passing" result on record validates code that no longer exists. There is currently no working automated proof that `web/lib/features.ts` matches `ml/src/features.py`.
2. **A5 (baseline models)** — `ml/src/baselines.py` doesn't exist; no comparison table for the thesis. Originally earmarked for the FYP student, but decided: implement now rather than wait.
3. **J3 (cross-machine benchmarks)** — only one stale (dummy-model-era) benchmark artifact exists; the automation script that was supposed to drive this (`ml/scripts/collect_benchmark.py`) targets a page structure deleted in the same merge.

The self-audit (against live repo, not just recollection) found three more things the first draft missed:

4. **J2 (end-to-end sanity check)** — `ml/scripts/e2e_app_test.py` is the third script the merge commit itself flagged as stale (alongside the two above), and `docs/PROGRESS.md:126` still lists it as an open item. Confirmed broken: it polls `window.__ENGINE_STATE.prediction` (doesn't exist anywhere in `web/`) and clicks `text=Start camera` (no such button exists — the current app auto-starts the camera). Missed in the first draft; added as Part 4 below.
5. **No CI exists anywhere in the repo.** `.github/workflows/` doesn't exist. BUILD_PLAN_1.md's J1 spec explicitly requires this ("wire it into CI so it can never silently regress") — without it, whatever J1 gate gets built can go stale exactly the way the current one did. `git remote -v` confirms the origin is GitHub, so GitHub Actions is the natural fit.
6. **Stale doc found in passing**: `docs/architecture.md:35` still says the landmarker uses "GPU delegate with CPU fallback." `docs/PROGRESS.md` (written the same day, later) says the GPU-first delegate was removed because it fails the J1 parity gate — the app is CPU-only now. `architecture.md` was never updated after that decision. Small fix, bundled into Part 1 since it's directly about the parity gate's own reasoning.

Also checked: the face-mesh/prediction cadence question — already correct (`usePipeline.ts:15`, `INFERENCE_STRIDE = 30` → one prediction per 3.0s window, per CONTRACT.md §6 Amendment 1). No change needed there. Any new e2e/benchmark automation below must wait **more than 3s** for a prediction, not assume the old 2Hz cadence — `docs/PROGRESS.md:129` flags this explicitly as a trap for anyone rewriting these scripts.

Decisions locked in:
- Implement A5 now, not deferred to the student. Report **both Validation and Test splits** (not Test-only) so it's directly comparable row-by-row against the existing `metrics_validation.csv`/`metrics_test.csv` from A8.
- J1's new tolerance stays at **0.02** (the value the team already empirically validated with the old harness), formalized as **CONTRACT.md Amendment 2** rather than left as an undocumented magic number.
- J3: automate + run once on the available dev machine, and leave a runbook for the other two machines rather than fabricate results for hardware not available here.
- CI: add a GitHub Actions workflow running everything below on push/PR, so this doesn't rot a second time.

---

## Preflight check (do this before writing any Part 1 code)

Verify `web/public/models/face_landmarker.task` is actually staged. It's gitignored, `createLandmarker()` (`lib/faceLandmarker.ts`) points `modelAssetPath` at `/models/face_landmarker.task`, but `scripts/copy-assets.mjs` (the `postinstall` hook) only copies WASM/mjs assets — not this file. Either there's a manual/undocumented step that puts it there, or the production app itself is currently missing it. If it's missing, Playwright's `webServer` wait in Part 1 will hang or time out looking like a mysterious CI failure instead of an obvious "model 404." Confirm this first; if missing, fix `copy-assets.mjs` (or document the manual download step from the README more prominently) as a precondition, not a surprise mid-implementation.

---

## Part 1 — J1: rebuild the feature-parity gate

**Root cause:** `web/harness/parity.html` imports `/src/features.js` (deleted) and calls the old `computeFeatures(pixels, frameShape, pitchCentre)` signature. The current `web/lib/features.ts` signature is `computeFeatures(landmarks, frameWidth, frameHeight, pitchCentre)` and takes **normalized** landmarks directly (pixel conversion happens inside). The old Python-driven runner (`ml/scripts/browser_tests.py`) also discovered headless Chromium won't present seeked `<video>` frames reliably — it only worked in headed mode, which isn't CI-friable.

**Design: replace video-seeking with pre-extracted static frames**, which sidesteps the headless-seek bug entirely and lets this run as a normal headless Playwright test.

1. **Extend `ml/scripts/make_parity_fixture.py`** to also dump the sampled frames as PNGs (`ml/tests/fixtures/parity_frames/frame_###.png`), using the same `sample_step`/`fps` logic already used to build `parity_expected.json`, via `ffmpeg -vf select=...`. These stay gitignored/shared-directly like the existing video fixtures (DAiSEE license). Fix the file's stale mp4-vs-webm docstring while touching it.

2. **Build the test page as a real Next.js route**, e.g. `web/app/__parity-test__/page.tsx` — **not** a standalone static HTML file. The old harness imported MediaPipe via an absolute `/node_modules/...` path served by a raw Python `http.server`; under Playwright's `webServer` running `next dev`/`next start`, Next never exposes `node_modules` as static files, so that import would 404. A real route goes through the normal bundler instead.
   - **Reuse `createLandmarker()` from `web/lib/faceLandmarker.ts`** rather than hand-configuring a second `FaceLandmarker` instance in the test page. This is the exact factory production uses (CPU delegate, correct model path, `numFaces`), so the parity gate always tests production's actual config — a future change to landmarker options then can't silently drift out of what J1 validates, which is what happened to the old harness.
   - Add `web/playwright.config.ts`, scoped to `web/tests/e2e/` (or a `**/*.parity.test.ts` testMatch) so it doesn't collide with vitest's existing `include: ['tests/**/*.test.ts']` (confirmed this glob would otherwise pick it up). Update `web/vitest.config.ts` to exclude it.

3. **Write `web/tests/e2e/features.parity.test.ts`** (real Playwright Test, TS):
   - Navigates to the `__parity-test__` route, which for each pre-extracted PNG frame: draws it to a canvas, runs `detectForVideo` via the reused `createLandmarker()` instance, takes the primary face's landmarks, calls `computeFeatures(landmarks, width, height, pitchCentre)` from `lib/features.ts` directly.
   - Fetches `parity_expected.json`, compares per-feature max-abs-diff across all frames against the **0.02** tolerance, using the same face-mismatch/coverage gates the old harness already validated (`compared >= n*0.95`, `faceMismatch <= ceil(n*0.02)`).
   - Writes a fresh `docs/results/parity_report.json` on run, replacing the stale one.

4. **Retire the stale scripts**: delete `web/harness/parity.html` and `ml/scripts/browser_tests.py`'s parity-driving portion. If `browser_tests.py` also does something still-useful (ONNX smoke test), keep only that part and note the split — don't block on it.

5. **Update `CONTRACT.md`**: add **Amendment 2** documenting the 0.02 parity tolerance and why (MediaPipe CPU-delegate Python-vs-browser landmark noise, GPU delegate excluded). **Fix `docs/architecture.md:35`** ("GPU delegate with CPU fallback" → CPU-only, matching `PROGRESS.md` and reality). Update `docs/PROGRESS.md`'s open-items list to remove the stale-J1 entry once this lands.

6. **Wire into `package.json`**: add `"test:parity": "playwright test"` (kept separate from `"test": "vitest run"` since it's a different runner and needs a browser + built app).

7. **Add `.github/workflows/ci.yml`**: on push/PR, run (web) `npm ci && npm test && npm run test:parity` and (ml) `pip install -r requirements.txt && pytest ml/tests/`. This is the actual fix for "J1 must never silently regress" — a passing local run isn't enough, it has to run on every change automatically, which is the thing that didn't happen the first time.

---

## Part 2 — A5: baseline models

**New file `ml/src/baselines.py`**, matching existing conventions in `train.py`/`eval.py` (module docstring tagged "(A5)", `REPO_ROOT` sys.path boilerplate, argparse with `description=__doc__`):

1. Load `artifacts/dataset/{Train,Validation,Test}.npz` via the established pattern (`data["x"]`, `data["y_engagement"]`). Note `x` is already scaler-standardized — consistent with how train.py/eval.py consume it, so no separate raw-feature path needed.
2. For each window `(30, 13)`, compute a 65-dim aggregate vector: mean/std/min/max/range per feature (flatten across time). This logic doesn't exist anywhere yet — new code, not a port.
3. Train `LogisticRegression(class_weight="balanced", max_iter=...)` and `RandomForestClassifier(class_weight="balanced")` once on Train; evaluate on **both Validation and Test**, matching how A8 reports both (`metrics_validation.csv`, `metrics_test.csv`).
4. Reuse the existing majority-class idiom (`train.py:233-237` / `eval.py:134-138`: `np.bincount(y).argmax()` then macro-F1 against that constant prediction) rather than reinventing it.
5. Write `docs/results/baselines.csv` via `csv.writer`, same output-path convention as `eval.py` (`REPO_ROOT / "docs" / "results"`, `mkdir(parents=True, exist_ok=True)`). Columns: `split, model, macro_f1, accuracy` — the `split` column is what makes this directly diffable against the TCN's own per-split CSVs. Optionally add the same "3-class merged (0+1=low) macro-F1" row `metrics_test.csv` already reports, for a cleaner side-by-side thesis table (nice-to-have, not blocking).
6. Run it once for real to produce the actual CSV (a real result, not a stub) — report the numbers honestly, same principle A7/A8 already followed.

---

## Part 3 — B7/J3: benchmark refresh + multi-machine runbook

1. **Add hardware-hint fields directly to `web/lib/benchmark.ts`**: extend `runBenchmark()`'s returned `BenchmarkResult` with `hardwareConcurrency: navigator.hardwareConcurrency` and `deviceMemory: (navigator as any).deviceMemory ?? null`. These are `navigator.*` properties only readable from inside the page — they can't be usefully injected after the fact by an external Playwright driver, so they belong in the source object itself. This makes both the manual browser-button download and the Playwright-collected JSON carry the same shape automatically.
2. **Rewrite `ml/scripts/collect_benchmark.py`** against the current app (confirmed stale: it targets a deleted `/bench` page, `window.__BENCH`, and button text "Run benchmark" that no longer exist). New version:
   - Targets the current main page and its actual button text ("Run 300 inferences") from `BenchmarkPanel.tsx`.
   - Since the current flow triggers a file **download** (not a `window.__BENCH` global), handle Playwright's `page.on("download")` event, save the downloaded JSON, and copy it into `docs/benchmarks/`.
   - Accept a `--machine-label` CLI arg (e.g. `"dev-laptop-i7-16GB"`) used in the output filename — CPU model/RAM amount still need to be human-supplied in that label, since `hardwareConcurrency`/`deviceMemory` (added in step 1) are rough proxies at best, not exact hardware identification.
3. **Run it now** on this dev machine against the real (already-shipped) `model_int8.onnx`, producing a fresh, correctly-labeled artifact in `docs/benchmarks/` — replacing reliance on the stale dummy-model file.
4. **Write a short `docs/benchmarks/README.md` runbook**: how to run `collect_benchmark.py --machine-label "..."` on another machine (needs Node + built `web/` app + Python + Playwright), and where to drop the resulting JSON. This is what makes the other 2 required machines actionable without fabricating numbers for hardware not available here.

---

## Part 4 — J2: fix the end-to-end sanity-check script

`ml/scripts/e2e_app_test.py` is the third script the merge commit flagged as stale, missed in the first draft of this plan. It's a fake-webcam-device end-to-end test (BUILD_PLAN_1.md's J2: "eyes closed for 3 seconds should visibly move the engagement output") — distinct from J1 (feature math parity) and B7 (performance benchmarking). Confirmed broken on two counts: it polls a `window.__ENGINE_STATE.prediction` global that doesn't exist, and clicks a `text=Start camera` button that doesn't exist (the current app auto-starts the camera via `usePipeline.ts`/`CognitiveApp.tsx`).

1. **Expose a minimal debug hook from `usePipeline.ts`**: mirror `status`/`prediction` onto `window.__ENGINE_STATE` (or similar) so an external Playwright driver can poll real app state without scraping the DOM. Keep it small — just what the e2e script needs to assert "a prediction arrived and changed."
2. **Retarget the script**: drop the `Start camera` click (nothing to click — camera starts automatically once permission is granted for the fake device), wait on the new debug hook instead of the old global.
3. **Respect the 3s cadence**: the wait timeout for "a prediction arrived" must be comfortably over 3 seconds (e.g. 8-10s), not the old 2Hz-era assumption — this is explicitly called out in `docs/PROGRESS.md:129` as a trap for whoever fixes this script.
4. Re-run to refresh `docs/results/app_e2e.json` and `app_screenshot.png` (existing artifact names already referenced in `docs/PROGRESS.md`'s artifact index — this is a refresh of a known artifact type against the merged app, not a new one).

---

## Files touched (summary)

- `web/public/models/face_landmarker.task` staging — verified/fixed if missing (preflight)
- `ml/scripts/make_parity_fixture.py` — extend to dump PNG frames, fix stale docstring
- `web/app/__parity-test__/page.tsx` — new, reuses `createLandmarker()`
- `web/playwright.config.ts` — new
- `web/vitest.config.ts` — exclude parity test
- `web/tests/e2e/features.parity.test.ts` — new, the real J1 gate
- `web/harness/parity.html` — retired
- `ml/scripts/browser_tests.py` — parity portion retired
- `CONTRACT.md` — Amendment 2 (parity tolerance)
- `docs/architecture.md` — fix stale GPU-delegate claim
- `docs/PROGRESS.md` — remove stale J1/J2 open-items
- `web/package.json` — add `test:parity` script
- `.github/workflows/ci.yml` — new
- `ml/src/baselines.py` — new
- `docs/results/baselines.csv` — new, real output (both splits)
- `web/lib/benchmark.ts` — add `hardwareConcurrency`/`deviceMemory` fields
- `ml/scripts/collect_benchmark.py` — rewritten against current app
- `docs/benchmarks/` — fresh real-model artifact + new `README.md` runbook
- `web/hooks/usePipeline.ts` — expose debug hook for e2e
- `ml/scripts/e2e_app_test.py` — rewritten against current app
- `docs/results/app_e2e.json`, `app_screenshot.png` — refreshed

## Verification

- J1: `npm run test:parity` (from `web/`) runs headless and passes/fails on real current code against `parity_expected.json` at 0.02 tolerance; confirm it fails loudly if someone reverts `lib/features.ts` to something wrong (sanity-check by temporarily breaking a formula and re-running).
- CI: push a branch, confirm `ci.yml` actually runs and goes green; then temporarily break something (e.g. a feature formula) in a throwaway commit and confirm CI goes red, before reverting — this proves the regression gate actually gates.
- A5: `python ml/src/baselines.py` runs clean, `docs/results/baselines.csv` has real, sane macro-F1 numbers for both splits (majority-class rows should match the existing majority-class numbers already recorded in `docs/results/metrics_{validation,test}.csv` from A8, as a cross-check).
- B7/J3: `python ml/scripts/collect_benchmark.py --machine-label dev-machine` produces a new JSON in `docs/benchmarks/` with the real int8 model's numbers, including the new hardware-hint fields (sanity-check backend/threads fields aren't the dummy-era values).
- J2: rewritten `ml/scripts/e2e_app_test.py` passes against a fake webcam device, waiting >3s for a prediction, and produces a fresh `app_e2e.json`/`app_screenshot.png`.
- Full `npm test` (vitest) and existing `pytest ml/tests/` still pass after these changes — nothing here should break existing coverage.
