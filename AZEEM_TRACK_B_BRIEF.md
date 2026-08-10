# Azeem — Track B Brief: Web & Edge Inference

**Project:** Privacy-preserving cognitive state detection running entirely in the browser
**Your remit:** the whole web application and in-browser inference layer
**Partner:** Bilal (Track A — data extraction, model training, ONNX export)
**Duration:** 20 days, ~3–4 hrs/day

This document is self-contained. You should not need to ask Bilal anything to start.

---

## 1. What the project is

Students on video calls give off non-verbal signals about whether they're following along. A model can read those signals — but streaming student webcam video to a server is a privacy disaster and probably illegal under GDPR.

So: we extract lightweight geometric features from the face locally (eye openness, mouth openness, head angle, gaze), run a tiny neural network **in the browser**, and never transmit a single video frame. Bilal trains the model on the DAiSEE dataset. You build the thing that runs it live.

The headline claims we have to prove are:
1. It runs at **≥30 FPS** on ordinary consumer hardware, no GPU required
2. **Zero bytes** of visual data leave the machine

Both of those are yours to demonstrate.

---

## 2. How we work in parallel

You are **not** waiting for Bilal's model. On Day 2 you generate a fake ONNX file with the correct input/output shape and build the entire app against it. When the real model arrives on Day 11, it's a file swap and nothing else changes.

This works only because of the **Interface Contract** in Section 3. It is frozen after Day 1. If something in it has to change, we both stop, agree, and update it in the same commit. No silent changes — a quiet edit to the feature order on one side is the single most expensive bug this project can have.

**Sync points (both of us, live):** Day 1, Day 6, Day 12, Day 15, Day 20.

---

## 3. THE INTERFACE CONTRACT

Copy of `CONTRACT.md`. Treat as law.

### 3.1 Feature vector — 13 floats per frame, order fixed

| # | Name | Definition |
|---|---|---|
| 0 | `ear_left` | Eye Aspect Ratio, left eye |
| 1 | `ear_right` | Eye Aspect Ratio, right eye |
| 2 | `ear_mean` | mean of 0 and 1 |
| 3 | `mar` | Mouth Aspect Ratio (vertical ÷ horizontal lip distance) |
| 4 | `brow_left` | left eyebrow-to-eye-centre distance ÷ interocular distance |
| 5 | `brow_right` | right eyebrow-to-eye-centre distance ÷ interocular distance |
| 6 | `yaw` | geometric yaw proxy, ~−1..1 |
| 7 | `pitch` | geometric pitch proxy, ~−1..1 |
| 8 | `roll` | true roll angle, radians |
| 9 | `gaze_x` | iris centre x offset from eye-corner midpoint ÷ eye width |
| 10 | `gaze_y` | iris centre y offset ÷ eye height |
| 11 | `face_area` | face bbox area ÷ frame area |
| 12 | `face_present` | 1.0 if landmarks detected, else 0.0 |

**Normalisation rule:** every distance is divided by the interocular distance (outer corner of left eye → outer corner of right eye) *before* anything else. This is what makes the features work identically whether the face is 30cm from a webcam or 1m from a DAiSEE recording. Do not skip it.

**Missing-face rule:** no face detected → emit thirteen zeros with `face_present = 0.0`. Never interpolate. Never skip the frame.

### 3.2 Head pose — read this, it affects you directly

The original plan called for `cv2.solvePnP`. **We are not doing that**, because there is no clean equivalent in the browser and pulling in opencv.js for one function would cost ~8 MB of WASM and wreck the edge story.

Instead, all three angles are pure landmark geometry — about ten lines, identical in Python and TypeScript:

```
roll  = atan2(rightEyeOuter.y - leftEyeOuter.y,
              rightEyeOuter.x - leftEyeOuter.x)

dL    = |noseTip - leftEyeOuter|          (2D)
dR    = |noseTip - rightEyeOuter|
yaw   = (dL - dR) / (dL + dR)

eyeMid = midpoint(leftEyeOuter, rightEyeOuter)
pitch  = ((noseTip.y - eyeMid.y) / |chin.y - eyeMid.y|) - PITCH_CENTRE
```

`PITCH_CENTRE` is a constant Bilal computes as the dataset mean; it lands in `scaler.json`. Until he ships it, use `0.5`.

These are proxies, not true Euler angles. That's fine and defensible — they're monotonic in the real angle, scale-invariant, and trivially portable. Just describe them accurately in the write-up.

**Do not** use MediaPipe's `outputFacialTransformationMatrixes` to shortcut this. It would give you different numbers from Bilal's and break parity silently.

### 3.3 Landmark indices

```
LEFT_EYE_EAR  = [33, 160, 158, 133, 153, 144]   // p1..p6, EAR order
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
MOUTH         = [61, 291, 13, 14]                // left, right, upper, lower
LEFT_BROW     = [70, 63, 105, 66, 107]
RIGHT_BROW    = [300, 293, 334, 296, 336]
INTEROCULAR   = [33, 263]                        // outer corners
NOSE_TIP      = 1
CHIN          = 152
LEFT_IRIS     = [468, 469, 470, 471, 472]
RIGHT_IRIS    = [473, 474, 475, 476, 477]
```

EAR formula: `(|p2−p6| + |p3−p5|) / (2 · |p1−p4|)`

These are the widely-used values but **verify them visually on Day 3** — draw the indices on a still frame and check each one sits where it should. Wrong indices give you a system that runs perfectly and means nothing.

### 3.4 Model I/O

```
INPUT   "features"     [1, 30, 13]  float32   (already standardised)
OUTPUT  "engagement"   [1, 4]       float32   logits → softmax → level 0..3
OUTPUT  "states"       [1, 4]       float32   logits → sigmoid → [bored, confused, engaged, frustrated]
```

Softmax and sigmoid happen in **your** JavaScript, not inside the graph.

### 3.5 Timing

- Sample the webcam at **10 FPS** (not 30 — the contract is 10, matching how the training video was sampled)
- Window = **30 frames = 3.0 seconds**
- Re-run inference every **5 frames** (every 0.5 s), sliding the buffer

Note the distinction for benchmarking: the *display* runs at 30+ FPS, the *feature sampling* runs at 10 Hz, and *inference* fires at 2 Hz. Report all three separately.

### 3.6 Standardisation

Bilal ships `public/model/scaler.json`:

```json
{ "mean": [13 floats], "std": [13 floats],
  "feature_names": [13 strings], "pitch_centre": 0.5, "version": "1.0" }
```

Apply `(x - mean) / std` element-wise before building the tensor. On load, assert `feature_names` matches your feature order and **throw** on mismatch — a loud crash beats silent garbage.

---

## 4. Your tasks

### B1 — Scaffold + dummy model · Day 2

Next.js 14 App Router, TypeScript, Tailwind. Install `onnxruntime-web` and `@mediapipe/tasks-vision`.

Generate the dummy model so you're never blocked (run once, commit the output):

```python
# scripts/make_dummy_onnx.py
import torch, torch.nn as nn
class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear(30*13, 64); self.a = nn.Linear(64,4); self.b = nn.Linear(64,4)
    def forward(self, x):
        h = torch.relu(self.f(x.flatten(1))); return self.a(h), self.b(h)
torch.onnx.export(Dummy(), torch.randn(1,30,13),
    "web/public/model/model_int8.onnx",
    input_names=["features"], output_names=["engagement","states"],
    opset_version=17)
```

Placeholder `scaler.json`: thirteen zeros for mean, thirteen ones for std.

`next.config.js` — without these headers WASM silently drops to single-threaded and your FPS numbers will be bad for no reason:

```js
module.exports = {
  headers: async () => [{
    source: '/(.*)',
    headers: [
      { key: 'Cross-Origin-Opener-Policy',   value: 'same-origin' },
      { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
    ],
  }],
};
```

**Done when:** dev server runs; dummy model loads in `onnxruntime-web`; a random `[1,30,13]` tensor returns two `[1,4]` outputs.

### B2 — Webcam · Day 3

`components/WebcamFeed.tsx`. `getUserMedia({ video: { width: 640, height: 480, frameRate: 30 } })`.

Handle these as real UI states, not crashes: permission denied, no camera present, camera in use by another app.

**Done when:** feed renders; denial shows a clear message; unmounting actually turns the camera indicator light off.

### B3 — MediaPipe · Days 4–5

`lib/faceLandmarker.ts`:

```ts
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const fileset = await FilesetResolver.forVisionTasks(
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
);
const landmarker = await FaceLandmarker.createFromOptions(fileset, {
  baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "GPU" },
  runningMode: "VIDEO",
  numFaces: 1,
  outputFaceBlendshapes: false,
});

const result = landmarker.detectForVideo(videoEl, performance.now());
const lm = result.faceLandmarks[0];  // array of {x, y, z}, normalised 0..1
```

**Assert `lm.length === 478` and log it.** If you get 468, the iris landmarks are missing and features 9–10 are broken. Download `face_landmarker.task` and self-host it rather than hotlinking — you want the demo to work if the venue wifi is bad.

Throttle detection to 10 Hz per the contract. Render landmarks in `LandmarkOverlay.tsx`.

**Done when:** 478 landmarks confirmed; overlay tracks the face; stable 10 Hz.

### B4 — Feature port · Day 5

`lib/features.ts` — a **line-for-line port** of Bilal's `ml/src/features.py`.

This is where projects like this die. Port it mechanically: same function names, same variable names, same order of operations. Do not tidy anything up. Do not vectorise it. Do not "fix" something that looks inefficient. Every deviation is a chance for the numbers to drift, and a drift of 0.01 produces an app that runs beautifully and predicts nothing.

Watch specifically for:
- MediaPipe JS gives **normalised** coords (0–1); Python may be using pixels. Multiply by frame width/height to match Bilal exactly.
- Radians vs degrees on `roll`
- Which eye is "left" — image-left or subject-left? Pick one, write it in the contract.

**Done when:** J1 passes. Not before.

### B5 — Ring buffer + inference · Days 7–8

`lib/ringBuffer.ts` — fixed `Float32Array(30 * 13)`, push-and-evict, `isFull()` guard so you never infer on a partial window.

`lib/inference.ts`:

```ts
import * as ort from 'onnxruntime-web';

ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4;
ort.env.wasm.simd = true;

// create ONCE at mount — never per frame
const session = await ort.InferenceSession.create('/model/model_int8.onnx', {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});

const tensor = new ort.Tensor('float32', standardised, [1, 30, 13]);
const out = await session.run({ features: tensor });
const engagement = softmax(out.engagement.data as Float32Array);
const states     = sigmoid(out.states.data as Float32Array);
```

**Done when:** predictions update ~2×/sec against the dummy model; JS heap flat over a 5-minute run in DevTools Memory.

### B6 — UI · Day 9

`PredictionPanel.tsx` — current engagement level, four state probability bars, rolling 60-second sparkline.

`PerfHUD.tsx` — live render FPS, ms/inference (rolling mean of last 30), model file size, active backend.

This is what the panel actually looks at. Dark background, one accent colour, large numbers readable from the back of a room on a projector. Please don't ship default-Tailwind grey boxes.

**Done when:** updates live, nothing flickers, readable across a room.

### B7 — Benchmark harness · Day 10

`lib/benchmark.ts` — a button that runs 300 inference cycles and reports mean / p50 / p95 / p99 ms, mean FPS, JS heap delta. Export as JSON to `docs/benchmarks/`.

**Done when:** three consecutive runs produce stable numbers.

### B8 — Real model · Days 12–14

Swap in Bilal's `model_int8.onnx` and real `scaler.json`. Validate `feature_names` on load.

Then benchmark fp32 vs int8 side by side — that comparison is a thesis result.

**Done when:** predictions respond sensibly to you closing your eyes, looking away, leaning back. If they don't, the problem is features or model, not UI. Go back to J1 before touching anything else.

### B9 — Privacy proof · Day 18

This has to be *proven*, not asserted:

- DevTools Network tab, XHR/Fetch filter, recorded over a full 60-second session, showing zero outbound requests after initial asset load → screenshot
- Code walkthrough showing frame `ImageData` is function-scoped and never persisted
- **Strongest version:** serve the demo page with `Content-Security-Policy: connect-src 'none'`. If the app still works with all network egress blocked at the policy level, that's a far better proof than any screenshot.

**Done when:** `docs/privacy.md` has the evidence and the written argument.

---

## 5. J1 — the parity gate · Day 6 · BLOCKING

Nothing downstream is trustworthy until this passes.

Bilal gives you `ml/tests/fixtures/parity_clip.mp4` and `parity_expected.json` (100 frames × 13 features, from the Python pipeline). You write `web/tests/features.parity.test.ts` that decodes the same file in headless Chrome via Playwright, runs `lib/features.ts`, and asserts **max absolute difference < 1e-4** per feature. Then wire it into CI.

**It will fail the first time.** Budget the whole day. Usual causes, in order of frequency:

1. Normalised vs pixel landmark coordinates
2. BGR vs RGB
3. Degrees vs radians
4. Frame sampling offset — Python takes frames 0,3,6…, you take 1,4,7…
5. 468 vs 478 landmarks (iris missing)
6. Left/right eye convention flipped

Debug it feature-by-feature, not all at once. Print the per-feature max diff and fix the worst one first.

---

## 6. Your timeline

| Day | Task |
|---|---|
| 1 | **JOINT** — repo, contract sign-off, environment |
| 2 | B1 scaffold + dummy ONNX |
| 3 | B2 webcam + verify landmark indices visually |
| 4 | B3 MediaPipe integration |
| 5 | B4 feature port |
| 6 | **J1 PARITY GATE — blocking** |
| 7 | B5 ring buffer |
| 8 | B5 inference session |
| 9 | B6 UI |
| 10 | B7 benchmark harness |
| 11 | Polish / cross-browser (Bilal exporting model) |
| 12 | **J2 INTEGRATION** |
| 13–14 | B8 real model, int8 vs fp32 |
| 15 | **J3 BENCHMARKS** (3 machines) |
| 16 | Buffer — assume you'll need it |
| 17 | Cross-browser check (Chrome, Edge, Firefox, Safari if possible) |
| 18 | B9 privacy proof |
| 19 | README + architecture diagram |
| 20 | **J4 FREEZE + DRY RUN** |

---

## 7. What you need from Bilal, and when

| When | What |
|---|---|
| Day 1 | Signed `CONTRACT.md` in the repo |
| Day 5 | `ml/src/features.py`, final version, for the port |
| Day 6 | `parity_clip.mp4` + `parity_expected.json` |
| Day 11 | `model_int8.onnx` + real `scaler.json` |
| Day 14 | `model_fp32.onnx` for the quantization comparison |

If any of these slip, keep going on the dummy model. Do not idle.

**Message him immediately** — don't wait for a sync — if: parity fails on a feature you can't explain, the contract needs changing, or the real model's outputs look wrong in a way that isn't a UI bug.

---

## 8. Using Claude Code

One task ID per session. Fresh session for each.

- **Paste `CONTRACT.md` first, every session.** It's short and it's what stops drift.
- **Ask for tests first:** "write the parity test, then write `features.ts` to pass it."
- **For B4, give it both sides:** paste the complete `features.py` and say *port this to TypeScript function by function, preserving names and operation order, change nothing else*.
- **Never let it invent numbers.** Landmark indices, feature order, tensor shapes all come from the contract. If it produces an index you didn't give it, verify visually before trusting it.
- End each session by asking for a summary of files changed and assumptions made. Paste that into the next session.

Suggested opening prompt for B1:

> I'm building a Next.js 14 App Router app in TypeScript that runs ONNX inference in-browser via onnxruntime-web, with MediaPipe Tasks Vision for face landmarks. Here is the interface contract: [paste CONTRACT.md]. For this session only, set up the project scaffold: package.json, next.config.js with COOP/COEP headers for WASM threading, Tailwind, and a minimal page that loads an ONNX model from /model/model_int8.onnx and runs one random [1,30,13] tensor through it, logging both output shapes. Do not build the webcam or feature code yet.

---

## 9. Failure modes specific to your side

| Symptom | Likely cause |
|---|---|
| Predictions look random | Feature parity — go to J1 |
| Inference 10× slower than expected | COOP/COEP headers missing, WASM single-threaded |
| Features 9–10 always zero | 468 landmarks, iris model not loaded |
| Memory climbs steadily | Session or tensor recreated per frame |
| Works on your laptop, dies on the lab PC | GPU delegate unavailable — add CPU fallback |
| Predictions frozen | Ring buffer never fills; check the `isFull()` guard |
| Demo dies at the venue | CDN dependency — self-host the `.task` and WASM files |

---

## 10. Definition of done for Track B

- [ ] J1 parity test in CI and passing
- [ ] Live demo runs from clean `npm install && npm run build && npm start`
- [ ] ≥30 FPS render, inference latency recorded on three machines with specs
- [ ] fp32 vs int8 benchmark comparison
- [ ] `docs/privacy.md` with network evidence
- [ ] Survives: bad lighting, no face, glasses, two faces in frame
- [ ] Works in Chrome and Edge at minimum
- [ ] README with setup that works on a machine that isn't yours
