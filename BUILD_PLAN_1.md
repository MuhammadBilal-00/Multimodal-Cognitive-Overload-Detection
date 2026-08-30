# Build Plan — Privacy-Preserving Cognitive State Detection on Edge

**Team:** Bilal (Track A — Data & Model), Azeem (Track B — Web & Edge)
**Duration:** 20 days
**Working assumption:** 3–4 hrs/day each

> **Status note (2026-08-09):** this is the original Day-1 plan and is kept
> as written for the historical record — it is not updated day-to-day.
> `docs/PROGRESS.md` is the authoritative current-status log. As of
> 2026-08-09: the plan executed through Day 15 by 2026-08-03 (repository
> merge), after which a self-audit (`GAP_CLOSURE_PLAN.md`) found that the
> merge had left J1's harness, J2's e2e script, and J3's benchmark
> collector all silently stale (pointing at code/routes the merge itself
> deleted) and A5 baselines still undone. All four are now fixed — see the
> inline status notes on A5/J1/J2/J3 below and the Definition of Done in
> §10. Rebuilding J1 properly also caught and fixed a real feature-parity
> regression (`numFaces:4` multi-face support vs. the model's
> `numFaces:1` training data) — see CONTRACT.md Amendment 2.

---

## 0. How this plan works

The two tracks are designed to run **fully in parallel with zero blocking**. That is only possible because of one thing: the **Interface Contract** in Section 2. Both tracks build against the contract, not against each other's code.

Azeem does **not** wait for a trained model. He builds against a randomly-initialised ONNX file with the correct input/output shape from Day 2. When Bilal's real model lands on Day 11, it is a file swap.

**Rule:** the contract is frozen after Day 1. If it must change, both people stop, agree the change, and update `CONTRACT.md` in the same commit. No silent changes.

---

## 1. Roles and split rationale

| | Bilal (Track A) | Azeem (Track B) |
|---|---|---|
| Owns | Python, extraction, features, training, ONNX export | Next.js, TypeScript, in-browser inference, benchmarking |
| Deliverable | `model_int8.onnx` + `scaler.json` + metrics | Working web app hitting ≥30 FPS |
| Language | Python 3.11 | TypeScript / React |

Swap these if Azeem is the stronger Python person — the split is by pipeline boundary, not by skill, so it works either way. What matters is that **one person owns each side of the contract**.

**Joint days (both, same room or same call):** Day 1, Day 6, Day 12, Day 15, Day 20. These are integration gates. Do not skip them.

---

## 2. THE INTERFACE CONTRACT

This is the single most important section. Copy it into `CONTRACT.md` at the repo root on Day 1 and both sign off.

### 2.1 Feature vector — 13 floats per frame

Computed identically in Python (training) and TypeScript (inference). Order is fixed and must never change.

| # | Name | Definition | Range |
|---|---|---|---|
| 0 | `ear_left` | Eye Aspect Ratio, left eye | ~0.0–0.45 |
| 1 | `ear_right` | Eye Aspect Ratio, right eye | ~0.0–0.45 |
| 2 | `ear_mean` | mean of 0 and 1 | ~0.0–0.45 |
| 3 | `mar` | Mouth Aspect Ratio (vertical / horizontal lip distance) | ~0.0–1.0 |
| 4 | `brow_left` | left eyebrow-to-eye-centre distance ÷ interocular distance | ~0.0–0.6 |
| 5 | `brow_right` | right eyebrow-to-eye-centre distance ÷ interocular distance | ~0.0–0.6 |
| 6 | `yaw` | head rotation, radians | −π/2 to π/2 |
| 7 | `pitch` | head rotation, radians | −π/2 to π/2 |
| 8 | `roll` | head rotation, radians | −π/2 to π/2 |
| 9 | `gaze_x` | iris centre x offset from eye-corner midpoint ÷ eye width | ~−1.0–1.0 |
| 10 | `gaze_y` | iris centre y offset ÷ eye height | ~−1.0–1.0 |
| 11 | `face_area` | face bbox area ÷ frame area | 0.0–1.0 |
| 12 | `face_present` | 1.0 if landmarks detected, else 0.0 | 0 or 1 |

**Normalisation rule:** every distance is divided by the interocular distance (outer corner of left eye to outer corner of right eye) **before** any other processing. This makes features invariant to how close the face is to the camera. Non-negotiable — without it the browser (face close to webcam) and DAiSEE (face further away) produce incompatible values.

**Missing-face rule:** if MediaPipe returns no face, emit `[0,0,0,0,0,0,0,0,0,0,0,0,0]` with `face_present = 0.0`. Never interpolate, never drop the frame. The model learns to handle it.

### 2.2 Landmark indices — VERIFY THESE

These are the commonly-used MediaPipe Face Mesh indices. **Do not trust them blind.** On Day 2, render the mesh with indices drawn on a still image and confirm each one visually. Wrong indices produce a model that trains fine and is silently meaningless.

```
LEFT_EYE_EAR  = [33, 160, 158, 133, 153, 144]    # p1..p6, EAR order
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
MOUTH         = [61, 291, 13, 14]                 # left, right, upper, lower
LEFT_BROW     = [70, 63, 105, 66, 107]
RIGHT_BROW    = [300, 293, 334, 296, 336]
INTEROCULAR   = [33, 263]                          # outer corners
LEFT_IRIS     = [468, 469, 470, 471, 472]          # requires refine_landmarks=True
RIGHT_IRIS    = [473, 474, 475, 476, 477]
```

EAR formula: `(|p2−p6| + |p3−p5|) / (2 · |p1−p4|)`

### 2.3 Model I/O

```
INPUT   name: "features"    shape: [1, 30, 13]   dtype: float32
                            (batch, timesteps, features) — already standardised

OUTPUT  name: "engagement"  shape: [1, 4]        dtype: float32   logits, argmax → level 0..3
OUTPUT  name: "states"      shape: [1, 4]        dtype: float32   logits, sigmoid → [bored, confused, engaged, frustrated]
```

Softmax and sigmoid are applied **in JavaScript**, not in the graph. Keeps the ONNX graph minimal and the JS side explicit.

### 2.4 Sampling

- Video sampled at **10 FPS** (both training extraction and live browser capture)
- Window = **30 frames = 3.0 seconds**
- Live inference stride = **5 frames = 0.5 s** (ring buffer, slide by 5, re-infer)

### 2.5 Standardisation

Bilal exports `public/model/scaler.json`:

```json
{
  "mean": [13 floats],
  "std":  [13 floats],
  "feature_names": [13 strings],
  "version": "1.0"
}
```

Azeem applies `(x - mean) / std` element-wise in TS before building the tensor. Same numbers, both sides. `feature_names` exists so a mismatch fails loudly instead of silently.

---

## 3. Repository structure

```
cognitive-edge/
├── CONTRACT.md                  # Section 2 of this doc — frozen
├── BUILD_PLAN.md                # this file
├── README.md
├── .gitignore                   # MUST exclude DAiSEE video + extracted frames
│
├── ml/                          # ── TRACK A (Bilal)
│   ├── requirements.txt
│   ├── config.yaml              # all paths, hyperparams — no magic numbers in code
│   ├── src/
│   │   ├── features.py          # THE reference implementation of Section 2.1
│   │   ├── extract.py           # multiprocess video → per-clip CSV
│   │   ├── labels.py            # DAiSEE label CSVs → clean dataframe
│   │   ├── dataset.py           # CSVs → windowed .npy tensors
│   │   ├── model.py             # TCN definition
│   │   ├── train.py
│   │   ├── evaluate.py          # confusion matrices, macro-F1, AUC, plots
│   │   ├── baselines.py         # logistic regression + random forest
│   │   └── export_onnx.py       # export + quantize + verify
│   ├── tests/
│   │   ├── test_features.py
│   │   └── fixtures/
│   │       ├── parity_clip.mp4  # 10s fixture used by BOTH tracks
│   │       └── parity_expected.json
│   └── artifacts/               # gitignored except final model
│
├── web/                         # ── TRACK B (Azeem)
│   ├── package.json
│   ├── next.config.js           # COOP/COEP headers for WASM threads
│   ├── app/
│   │   ├── page.tsx
│   │   └── layout.tsx
│   ├── components/
│   │   ├── WebcamFeed.tsx
│   │   ├── LandmarkOverlay.tsx
│   │   ├── PredictionPanel.tsx
│   │   └── PerfHUD.tsx          # live FPS / ms-per-inference
│   ├── lib/
│   │   ├── features.ts          # PORT of ml/src/features.py — line-for-line
│   │   ├── faceLandmarker.ts    # MediaPipe Tasks Vision wrapper
│   │   ├── ringBuffer.ts
│   │   ├── inference.ts         # onnxruntime-web session
│   │   └── benchmark.ts
│   ├── public/model/
│   │   ├── model_int8.onnx
│   │   └── scaler.json
│   └── tests/
│       └── features.parity.test.ts
│
└── docs/
    ├── results/                 # figures for the thesis
    └── benchmarks/
```

`.gitignore` must include `*.mp4`, `*.avi`, `ml/data/`, `frames/`. The DAiSEE licence forbids redistributing the videos. Committing them is a licence breach and will also blow past GitHub's file limits.

---

## 4. TRACK A — Data & Model (Bilal)

### A1 — Environment and data acquisition · Day 1

Submit the DAiSEE Google Form (linked from `people.iith.ac.in/vineethnb/resources/daisee/index.html`), start the ~15 GB download immediately. Budget 60–80 GB free disk.

Set up Python 3.11 venv. Pin: `mediapipe`, `opencv-python`, `torch`, `pandas`, `numpy`, `scikit-learn`, `onnx`, `onnxruntime`, `matplotlib`, `pyyaml`, `tqdm`.

**Also on Day 1, run this and record the answer:**

```bash
ffprobe -v error -show_streams path/to/one_daisee_clip.avi | grep codec_type
```

If there is **no audio stream**, the "multimodal" claim must be redefined before anything is written. Fallback framing: fusion of *geometric* (EAR/MAR/brow), *pose* (yaw/pitch/roll) and *gaze* modalities — three distinct visual signal families, fused in a single network. That is still legitimately multimodal and defensible. Decide Day 1, not Day 15.

**Acceptance:** venv reproducible from `requirements.txt`; audio question answered in writing; download running.

### A2 — Feature module · Day 2

`ml/src/features.py` — pure functions, no I/O:

```python
def compute_features(landmarks: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """landmarks: (478, 3) or None. Returns (13,) float32 per CONTRACT.md §2.1"""
```

Sub-functions: `eye_aspect_ratio`, `mouth_aspect_ratio`, `brow_raise`, `head_pose`, `gaze_offset`, `face_area`.

Head pose via `cv2.solvePnP` against a canonical 3D face model using 6 stable points (nose tip, chin, both eye outer corners, both mouth corners), converted to Euler angles.

**Acceptance:** `pytest ml/tests/test_features.py` passes with hand-checked values for a synthetic open-eye and closed-eye landmark set. Indices visually verified against a rendered mesh.

### A3 — Extraction pipeline · Days 3–4

`ml/src/extract.py`. Walks the DAiSEE directory tree, and for each clip: opens with OpenCV, samples to 10 FPS, runs MediaPipe FaceMesh (`refine_landmarks=True`, `static_image_mode=False`), writes `artifacts/features/{clip_id}.csv` with 13 columns + `frame_idx`.

**Must use `multiprocessing.Pool`.** Serial is ~5–6 hours; 8 workers is ~45–90 min. Include `--resume` so a crash at clip 6000 doesn't restart from zero. Write a per-clip failure log rather than aborting the run.

**Acceptance:** ≥95% of the 9,068 clips produce a CSV. Failures logged with reasons. A `stats.json` reporting mean face-detection rate — if it's under 85%, stop and investigate before training on garbage.

### A4 — Labels and windowing · Day 5

`ml/src/labels.py` + `dataset.py`.

**Use DAiSEE's official Train / Validation / Test folders.** They are already subject-independent (5482 / 1723 / 1720). Do not re-split. This is both easier and makes results directly comparable to the published benchmark.

Label design:
- **Primary task:** 4-class engagement level (0–3) — the standard DAiSEE benchmark task
- **Secondary task:** 4 binary heads, one per state, thresholded at level ≥2

Windowing: stride 10 over each clip's ~100 rows → ~8 windows/clip. Clip label applied to every window from that clip. Output `X_train.npy (N,30,13)`, `y_eng_train.npy`, `y_states_train.npy`, same for val/test.

Fit `StandardScaler` **on train only**, apply to all three, save to `scaler.json`.

**Acceptance:** shapes printed and sane; zero clip-ID overlap between splits (assert it in code); class distribution per split written to `docs/results/class_dist.png`.

### A5 — Baselines · Day 5 (or hand to the FYP student)

`ml/src/baselines.py`. Flatten windows to per-window aggregate stats (mean, std, min, max, range of each of 13 features = 65 dims). Train logistic regression and random forest. Report macro-F1 and the majority-class score.

This gives the thesis its comparison table. It is also the single best task to hand to the student — an afternoon's work, and it is legitimately hers.

**Acceptance:** `docs/results/baselines.csv` with majority-class, LogReg, RF macro-F1.

> **Status (2026-08-09): DONE**, implemented directly rather than handed
> to the student (decision recorded in `GAP_CLOSURE_PLAN.md`). Reports
> both Validation and Test splits (extra scope vs. the acceptance
> criterion above, for row-by-row comparison against A8's own
> `metrics_{validation,test}.csv`). Majority-class rows cross-check
> exactly against those files (0.1813 / 0.1655). TCN val macro-F1
> (0.3061) beats both LogReg (0.242) and RF (0.2669).

### A6 — Model · Days 6–7

`ml/src/model.py` — small dilated TCN:

```
Input (B, 30, 13) → transpose → (B, 13, 30)
Block1: Conv1d(13→64, k=3, dilation=1, padding=causal) + BN + ReLU + Dropout(0.2)
Block2: Conv1d(64→64, k=3, dilation=2) + BN + ReLU + Dropout(0.2)
Block3: Conv1d(64→64, k=3, dilation=4) + BN + ReLU + Dropout(0.2)
        (receptive field = 15 timesteps; add dilation=8 block for full 30)
Residual connections around each block
GlobalAvgPool over time → (B, 64)
├── FC(64→4)  → engagement logits
└── FC(64→4)  → state logits
```

Target: **under 100k parameters**. The whole edge story depends on the model being tiny — keep it that way.

**Acceptance:** forward pass on random input returns correct shapes; parameter count printed and <100k.

### A7 — Training · Days 8–10

`ml/src/train.py`. Adam, LR 1e-3, cosine schedule, batch 128, max 100 epochs, early stopping on **val macro-F1** (not val loss, not accuracy) with patience 15.

**Class imbalance is the whole game here.** DAiSEE's low-engagement classes are roughly 0.7% and 5% of the data. Use `CrossEntropyLoss(weight=...)` with inverse-frequency weights. If macro-F1 is still poor, try focal loss (γ=2). A model that predicts "high engagement" for everything will show ~50% accuracy and near-zero macro-F1 — that is the failure mode to watch for.

Loss = `CE(engagement) + 0.5 · BCE(states)`.

Log every run to `artifacts/runs/{timestamp}/` with config, curves, best checkpoint. Set seeds and record them.

**Acceptance:** val macro-F1 beats the RF baseline from A5. Training curves saved. If it doesn't beat the baseline, that is a finding to report honestly, not a reason to fake it.

### A8 — Evaluation · Day 10, revisited Day 17

`ml/src/evaluate.py`. On the **test set, once, at the end**:

- Confusion matrix, raw and row-normalised → PNG
- Per-class precision / recall / F1 + macro and weighted averages
- One-vs-rest ROC curves and AUC
- The majority-class baseline printed alongside every number

**Acceptance:** everything in `docs/results/`, publication quality (labelled axes, readable font sizes).

### A9 — Export and quantize · Day 11

`ml/src/export_onnx.py`:

1. `torch.onnx.export` with `opset_version=17`, named inputs/outputs per §2.3, `dynamic_axes` on batch only
2. `onnx.checker.check_model`
3. **Parity check:** run 100 random inputs through PyTorch and through `onnxruntime`, assert max absolute difference < 1e-5. Fail the build otherwise.
4. Dynamic int8 quantization via `onnxruntime.quantization.quantize_dynamic`
5. Re-run test-set evaluation on the **quantized** model, record the accuracy delta
6. Copy `model_int8.onnx` + `scaler.json` to `web/public/model/`

**Acceptance:** both `.onnx` files exist; sizes recorded; fp32-vs-int8 macro-F1 delta recorded in `docs/results/quantization.csv`. Expect ~4× size reduction and a small F1 drop — that tradeoff table is a thesis result in itself.

---

## 5. TRACK B — Web & Edge (Azeem)

### B1 — Scaffold + dummy model · Day 2

Next.js 14 (App Router), TypeScript, Tailwind. Install `onnxruntime-web`, `@mediapipe/tasks-vision`.

**Generate a dummy ONNX now** so you are never blocked:

```python
# scripts/make_dummy_onnx.py — run once, commit the output
import torch, torch.nn as nn
class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(30*13, 64); self.a = nn.Linear(64,4); self.b = nn.Linear(64,4)
    def forward(self, x):
        h = torch.relu(self.f(x.flatten(1))); return self.a(h), self.b(h)
torch.onnx.export(Dummy(), torch.randn(1,30,13), "web/public/model/model_int8.onnx",
    input_names=["features"], output_names=["engagement","states"], opset_version=17)
```

Also commit a placeholder `scaler.json` of 13 zeros and 13 ones.

`next.config.js` needs COOP/COEP headers or WASM multithreading silently degrades to single-thread:

```js
headers: async () => [{ source: '/(.*)', headers: [
  { key: 'Cross-Origin-Opener-Policy',   value: 'same-origin' },
  { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' }]}]
```

**Acceptance:** `npm run dev` serves a page; dummy model loads in `onnxruntime-web` and returns two tensors of shape [1,4].

### B2 — Webcam · Day 3

`WebcamFeed.tsx` — `getUserMedia({ video: { width: 640, height: 480, frameRate: 30 } })`. Handle permission denial and no-camera with real UI states, not a crash. Stream into a `<video>` ref, draw to an offscreen canvas.

**Acceptance:** live feed renders; denying permission shows a clear message; `stop()` releases the camera on unmount (check the browser's camera indicator light actually goes off).

### B3 — MediaPipe in browser · Days 4–5

`lib/faceLandmarker.ts`. Load `FaceLandmarker` from CDN WASM with `outputFaceBlendshapes: false`, `numFaces: 1`, `runningMode: "VIDEO"`, and **`refineLandmarks` enabled** — without it you get 468 landmarks and no iris, and features 9–10 break.

Throttle to 10 FPS to match the contract. Draw landmarks on `LandmarkOverlay.tsx`.

**Acceptance:** 478 landmarks logged; overlay tracks the face; runs at a stable 10 Hz.

### B4 — Feature port · Day 5

`lib/features.ts` — a **line-for-line port** of `ml/src/features.py`.

This is where the project most commonly dies. Train/serve skew here produces an app that runs beautifully and predicts nonsense, and it is very hard to debug after the fact. Port it mechanically. Same variable names. Same order of operations. Do not "improve" anything.

**Acceptance:** B4 is not done until J1 passes.

### B5 — Ring buffer + inference · Days 7–8

`lib/ringBuffer.ts` — fixed 30×13 `Float32Array`, push-and-evict, `isFull()` guard.

`lib/inference.ts` — create `InferenceSession` once at mount (never per frame), WASM execution provider, apply scaler, build `ort.Tensor('float32', data, [1,30,13])`, run, apply softmax/sigmoid in JS, return typed result.

Run inference every 5 frames, not every frame.

**Acceptance:** predictions update ~2×/sec against the dummy model; no memory growth over 5 minutes (check DevTools Memory).

### B6 — UI · Day 9

`PredictionPanel.tsx` — current engagement level, 4 state probability bars, and a rolling 60-second sparkline of engagement.

`PerfHUD.tsx` — live FPS, ms/inference (rolling mean of last 30), model size, backend in use.

Design note: this is the demo the panel actually sees. Dark, clean, one accent colour, large readable numbers. Do not ship a default-Tailwind grey box.

**Acceptance:** panel updates live; nothing flickers; readable from across a room on a projector.

### B7 — Benchmark harness · Day 10

`lib/benchmark.ts` — a "Run Benchmark" button that captures 300 inference cycles and reports mean/p50/p95/p99 ms, mean FPS, JS heap delta, then exports the results as JSON to `docs/benchmarks/`.

**Acceptance:** produces a reproducible JSON; run it three times and confirm the numbers are stable.

### B8 — Real model integration · Days 12–14

Swap in Bilal's `model_int8.onnx` and real `scaler.json`. Validate `feature_names` in the scaler against the TS feature order at load time and throw loudly on mismatch.

Benchmark fp32 vs int8 side by side.

**Acceptance:** real predictions respond sensibly when you deliberately close your eyes, look away, or lean back. If they do not, go to J1 before touching anything else.

### B9 — Privacy proof · Day 18

Build the artifact that proves the privacy claim rather than asserting it:

- DevTools Network tab, filtered to XHR/Fetch, recorded across a full 60-second inference session, showing zero outbound requests after initial asset load → screenshot
- A code walkthrough showing the frame `ImageData` is function-scoped and never persisted
- Optionally: a `Content-Security-Policy: connect-src 'none'` header on the demo page. If the app still works with all network egress blocked by policy, that is a much stronger proof than a screenshot.

**Acceptance:** screenshots + written argument in `docs/privacy.md`.

---

## 6. Joint gates

### J1 — Feature parity test · Day 6 · BLOCKING

Nothing downstream is trustworthy until this passes.

1. Pick one 10-second DAiSEE clip → `ml/tests/fixtures/parity_clip.mp4`
2. Bilal runs the Python pipeline, dumps `parity_expected.json`: 100 frames × 13 features
3. Azeem writes `web/tests/features.parity.test.ts` that decodes the same file in a headless browser (Playwright), runs `lib/features.ts`, and asserts **max abs diff < 1e-4** per feature
4. Wire it into CI so it can never silently regress

**Expect this to fail the first time.** Common causes: BGR/RGB swap, normalised vs pixel landmark coordinates, degrees vs radians in head pose, different frame-sampling offsets, iris landmarks absent because `refineLandmarks` is off. Budget the full day.

> **Status (2026-08-09): DONE, rebuilt.** First implementation passed
> (2026-08-02, worst diff 0.0079) but the 2026-08-03 merge deleted a file
> `web/harness/parity.html` imported, leaving the recorded "PASS"
> silently validating nonexistent code — undetected until the
> 2026-08-09 self-audit. Rebuilt as a real Playwright Test
> (`web/tests/e2e/features.parity.test.ts`, `npm run test:parity`)
> against 100 static PNG frames instead of a seeked `<video>` (headless
> Chromium doesn't reliably present seeked frames). Tolerance loosened
> 1e-4 → 0.02 and formalized as CONTRACT.md Amendment 2 (the team's
> already-validated empirical value, not a new decision). Reusing the
> production landmarker factory (rather than the old harness's
> hand-configured second instance) caught a real regression the old
> harness structurally could never have caught: `numFaces:4` (added for
> multi-face UI after J1 first passed) fails this gate on blink frames;
> fixed via `createFeatureLandmarker()` (numFaces:1, feeds the model) vs.
> `createDisplayLandmarker()` (numFaces:4, overlay only). Wired into
> `.github/workflows/ci.yml` — runs unconditionally, except the actual
> parity test step, which needs the DAiSEE-derived fixture frames and so
> only runs on a runner that has them (never committed to git, per the
> dataset license — the same constraint that already kept the source
> clip out of git); skips with a visible warning otherwise. Verified the
> gate itself gates: deliberately broke a feature formula, confirmed
> `test:parity` failed loudly, reverted.

### J2 — End-to-end integration · Day 12

Real model, real features, live camera, full loop. Sanity check by behaviour: eyes closed for 3 seconds should visibly move the engagement output. If it doesn't, the model or the features are wrong — not the UI.

> **Status (2026-08-09): DONE, rebuilt.** First implementation passed
> (2026-08-02) but the 2026-08-03 merge deleted the app the script drove
> against — `ml/scripts/e2e_app_test.py` was polling a
> `window.__ENGINE_STATE` that no longer existed and clicking a "Start
> camera" button the merged app doesn't have (camera now auto-starts).
> Fixed by adding a minimal `window.__ENGINE_STATE` mirror
> (`{status, prediction, facePresent}`) to `hooks/usePipeline.ts` and
> retargeting the script at it; also fixed a leftover 2 Hz-era
> `time.sleep(2)` (now 4 s — must exceed the current 3 s inference
> window, CONTRACT.md §6 Amendment 1). Re-run against a fake webcam
> device: PASS, fresh `docs/results/app_e2e.json` / `app_screenshot.png`.

### J3 — Benchmark session · Day 15

Run B7 on at least three machines (both laptops plus one lab/library PC). Record CPU, RAM, browser version alongside every result. "≥30 FPS" is meaningless without stating the hardware.

> **Status (2026-08-09): PARTIAL — 1 of 3 machines.** The automation
> (`ml/scripts/collect_benchmark.py`) was also stale post-merge (targeted
> a deleted `/bench` page and `window.__BENCH`); rewritten against the
> current app's real "Run 300 inferences" button and its file-download
> flow, and re-run for real on this dev machine
> (`docs/benchmarks/benchmark-dev-i7-13700H-16GB.json`; `benchmark.ts`
> now also records `hardwareConcurrency`/`deviceMemory`). Two more
> machines are still needed — runbook in `docs/benchmarks/README.md` so
> this doesn't require fabricating numbers for hardware not available
> here.

### J4 — Freeze and dry run · Day 20

Tag `v1.0`. No commits after this. Full demo rehearsal including the failure cases — bad lighting, no face, two faces in frame. Know what breaks before the panel finds it.

---

## 7. Day-by-day timeline

| Day | Bilal (Track A) | Azeem (Track B) |
|---|---|---|
| 1 | **JOINT:** repo, CONTRACT.md, DAiSEE form + download, audio check | **JOINT:** same |
| 2 | A2 features.py + verify indices | B1 scaffold + dummy ONNX |
| 3 | A3 extraction script | B2 webcam |
| 4 | A3 run full extraction (long compute) | B3 MediaPipe |
| 5 | A4 labels + windowing, A5 baselines | B4 feature port |
| 6 | **J1 PARITY GATE** | **J1 PARITY GATE** |
| 7 | A6 model architecture | B5 ring buffer |
| 8 | A7 first training run | B5 inference session |
| 9 | A7 tuning, class weights | B6 UI |
| 10 | A7 finalise, A8 eval | B7 benchmark harness |
| 11 | A9 export + quantize → hand off | B7 polish, wait state |
| 12 | **J2 INTEGRATION** | **J2 INTEGRATION** |
| 13 | Debug skew / retrain if needed | B8 wire real model |
| 14 | Test-set eval on final model | B8 int8 vs fp32 |
| 15 | **J3 BENCHMARKS** | **J3 BENCHMARKS** |
| 16 | Buffer — *assume you need it* | Buffer / demo hardening |
| 17 | A8 final plots + baseline table | Cross-browser check |
| 18 | Support student's own runs | B9 privacy proof |
| 19 | README, methodology writeup | README, architecture diagram |
| 20 | **J4 FREEZE + DRY RUN** | **J4 FREEZE + DRY RUN** |

Day 16 is deliberately empty. Something will overrun — most likely J1 or training. If nothing does, use it to improve the demo UI, which is what the panel remembers.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Train/serve feature skew | **High** | J1 parity gate on Day 6, in CI |
| Model can't beat majority-class baseline | Medium | Class weighting, focal loss; if it still can't, **report it honestly** — a negative result properly analysed passes; a fabricated one fails |
| DAiSEE has no audio | Medium | Checked Day 1; fallback to 3 visual modality families |
| 15 GB download slow/fails | Medium | Start Day 1, use `wget -c` |
| MediaPipe browser ≠ MediaPipe Python versions | Medium | Pin both; note versions in CONTRACT.md |
| WASM threads disabled → slow inference | Medium | COOP/COEP headers in B1 |
| Face detection rate low on DAiSEE | Low-Med | A3 acceptance gate catches it before training |
| Wrong landmark indices | Low | Visual verification on Day 2 |

---

## 9. How to prompt Claude Code

The proposal's advice to compartmentalise is right. Practical version:

**Session structure — one task ID per session.** Open a fresh session for A3. Give it: `CONTRACT.md`, the A3 spec above, and `ml/src/features.py` (already working). Ask for `extract.py` only. Do not mention the web app.

**Lead every session with the contract.** Paste `CONTRACT.md` first, every time. It is short, and it is the thing that prevents drift.

**Demand the test first.** "Write `test_features.py` covering an open eye, a closed eye, and a `None` landmark input, then write `features.py` to pass it." Far better output than asking for the implementation alone.

**For the port (B4), give it both sides.** Paste the complete `features.py` and say: port this to TypeScript, function by function, preserving names and operation order, changing nothing else. Then run J1.

**Never let it invent the numbers.** Landmark indices, feature order, tensor shapes — these come from `CONTRACT.md`. If Claude produces an index you didn't specify, verify it visually before trusting it.

**Session hygiene:** end each session by asking for a summary of files changed and any assumptions made. Paste that summary into the next session's context.

---

## 10. Definition of done

- [ ] Repo tagged `v1.0`, README with setup instructions that work on a clean machine
- [x] J1 parity test in CI and passing — as of 2026-08-09, wired into `.github/workflows/ci.yml` and confirmed green there; caveat: the parity assertion itself runs only on a runner seeded with the DAiSEE-derived fixture (never committable per license), so a plain GitHub-hosted runner's job passes by correctly *skipping*, not by exercising the assertion. It has been exercised (and confirmed to catch a broken formula) locally.
- [x] `docs/results/` — confusion matrix, ROC curves, per-class metrics, baseline comparison table, class distribution — all present; `baselines.csv` (the comparison table) added 2026-08-09
- [ ] `docs/benchmarks/` — three machines, ms/inference, FPS, memory, hardware specs recorded — 1 of 3 (dev machine, 2026-08-09); runbook for the remaining two in `docs/benchmarks/README.md`
- [x] `docs/results/quantization.csv` — size and F1 for fp32 vs int8 — present, alongside `quantization_test.csv` (int8 61,650 B vs fp32 167,243 B; Test macro-F1 0.2460 vs 0.2475)
- [x] `docs/privacy.md` — network evidence + code argument — present; re-recorded 2026-08-29 against the production build under full CSP (`docs/results/privacy_trace.json`: 75 s, 39 requests, all same-origin)
- [x] Live demo runs from a clean `npm install && npm run build && npm start` — verified 2026-08-09 on a genuine fresh `git clone` (not this session's working copy); found and fixed one real gap (torch CPU wheel needs `--extra-index-url` up front, README previously implied it as a fallback-only step). Full record: `docs/demo-failure-modes.md`.
- [x] Demo survives: bad lighting, no face, glasses, two faces — verified 2026-08-09 with real DAiSEE clips (+ one synthetic two-face composite, since no natural DAiSEE clip has two simultaneously front-facing subjects) driven through a genuinely clean build via Playwright fake-camera injection. None crashed, froze, or produced a garbage prediction. Full record, including one UX observation (no dedicated "no face currently visible" banner): `docs/demo-failure-modes.md`.
- [x] DAiSEE citations in the README; no video files in git history — cited in README, and the history is clean (dataset clips and derived frames are gitignored under the DAiSEE licence)
- [ ] The student can walk the full pipeline unaided and answer the questions in her prep plan
