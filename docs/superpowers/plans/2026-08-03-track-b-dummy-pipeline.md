# Track B — Full Dummy-Model Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working Next.js web app that opens the webcam, extracts the 13 contract features from MediaPipe landmarks at 10 Hz, and runs a dummy ONNX model in-browser every 0.5 s, with live prediction UI, perf HUD, and benchmark harness (B1–B7 of the brief).

**Architecture:** Next.js 14 App Router app in `web/`. A rAF-driven pipeline: `<video>` (raw, un-mirrored) → MediaPipe FaceLandmarker (self-hosted WASM + `.task`, throttled to 10 Hz) → `features.ts` (contract §2–4, pixel coords) → 30×13 ring buffer → `onnxruntime-web` WASM session (created once) every 5th sample → softmax/sigmoid in JS → React state → dark dashboard UI. All assets self-hosted; COOP/COEP headers enable WASM threads.

**Tech Stack:** Next.js 14 (App Router, TS, Tailwind), onnxruntime-web, @mediapipe/tasks-vision, vitest for unit tests, Python `onnx` package for dummy model generation.

## Global Constraints

- Feature order FROZEN: `ear_left, ear_right, ear_mean, mar, brow_left, brow_right, yaw, pitch, roll, gaze_x, gaze_y, face_area, face_present` (CONTRACT §2).
- Landmark indices FROZEN exactly as CONTRACT §4. Never invent indices.
- Coordinates are **PIXELS**: `x_px = x_norm · frame_width`, `y_px = y_norm · frame_height` before any feature math (CONTRACT §4 conventions).
- "Left" = **IMAGE-LEFT** in an un-mirrored frame. Feed raw video to the landmarker; mirror only via CSS for display.
- Missing face → thirteen zeros, `face_present = 0.0`. Never interpolate, never skip.
- Head pose = geometric proxies (CONTRACT §3), epsilon `1e-8`, roll in radians, `PITCH_CENTRE` from scaler.json (placeholder 0.5). No `outputFacialTransformationMatrixes`.
- Model I/O: input `features` `[1,30,13]` f32; outputs `engagement` `[1,4]`, `states` `[1,4]` raw logits. Softmax/sigmoid in JS only. Standardise `(x−mean)/std` before tensor build.
- Sampling: features at 10 FPS, window 30 frames, inference every 5 frames (2 Hz). Display rAF unthrottled (30+ FPS).
- On scaler load, assert `feature_names` matches feature order; **throw** on mismatch.
- Self-host everything: `face_landmarker.task`, MediaPipe WASM, ort WASM. No CDN at runtime.
- COOP `same-origin` + COEP `require-corp` headers on all routes (WASM threading).
- ONNX session created ONCE at mount, never per frame. `isFull()` guard before inference.
- Parity interpretation points (brow distance definition, gaze eye-height definition, face bbox source) are marked `// J1-CHECK` in code — re-verify against Bilal's `features.py` on Day 5/6.

---

### Task 1: Next.js scaffold with COOP/COEP + deps (B1 part 1)

**Files:**
- Create: `web/` via create-next-app (package.json, app/, tailwind config, etc.)
- Modify: `web/next.config.mjs` (headers)
- Create: `web/vitest.config.ts`

**Interfaces:**
- Produces: running dev server on `http://localhost:3000` with COOP/COEP; `npm test` runs vitest; deps `onnxruntime-web`, `@mediapipe/tasks-vision` installed.

- [ ] **Step 1: Scaffold**

```powershell
cd C:\cognitive\Multimodal-Cognitive-Overload-Detection
npx create-next-app@14 web --typescript --tailwind --eslint --app --no-src-dir --import-alias "@/*" --use-npm
cd web
npm install onnxruntime-web @mediapipe/tasks-vision
npm install -D vitest
```

- [ ] **Step 2: COOP/COEP headers** — replace the config file (create-next-app@14 emits `next.config.mjs`):

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  headers: async () => [
    {
      source: '/(.*)',
      headers: [
        { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
        { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
      ],
    },
  ],
};
export default nextConfig;
```

- [ ] **Step 3: vitest config + test script**

`web/vitest.config.ts`:
```ts
import { defineConfig } from 'vitest/config';
export default defineConfig({ test: { include: ['tests/**/*.test.ts'] } });
```
Add to `web/package.json` scripts: `"test": "vitest run"`.

- [ ] **Step 4: Verify** — `npm run dev` starts; `curl -I http://localhost:3000` shows both headers. Kill server.

- [ ] **Step 5: Commit** — `git add web && git commit -m "feat(web): Next.js 14 scaffold with COOP/COEP headers and deps"`

---

### Task 2: Dummy ONNX + placeholder scaler.json (B1 part 2)

**Files:**
- Create: `scripts/make_dummy_onnx.py`
- Create: `web/public/model/model_int8.onnx` (generated, committed)
- Create: `web/public/model/scaler.json`

**Interfaces:**
- Produces: ONNX graph — input `features [1,30,13]` f32; outputs `engagement [1,4]`, `states [1,4]` f32 logits. `scaler.json` per CONTRACT §7 (mean=13×0, std=13×1, `pitch_centre: 0.5`).

- [ ] **Step 1: Write generator** (no torch on this machine — build the graph directly with the `onnx` package; identical I/O signature to the brief's torch script):

```python
# scripts/make_dummy_onnx.py
"""Dummy ONNX with the contract I/O signature (CONTRACT.md section 5).
features [1,30,13] -> flatten -> 64 relu -> engagement [1,4], states [1,4] (raw logits).
Run once, commit output: python scripts/make_dummy_onnx.py
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

rng = np.random.default_rng(42)
def w(name, shape, scale=0.1):
    return numpy_helper.from_array((rng.standard_normal(shape) * scale).astype(np.float32), name)

inits = [
    numpy_helper.from_array(np.array([1, 390], dtype=np.int64), "flat_shape"),
    w("W1", (390, 64)), w("b1", (64,)),
    w("W2", (64, 4)),  w("b2", (4,)),
    w("W3", (64, 4)),  w("b3", (4,)),
]
nodes = [
    helper.make_node("Reshape", ["features", "flat_shape"], ["flat"]),
    helper.make_node("Gemm", ["flat", "W1", "b1"], ["h_pre"]),
    helper.make_node("Relu", ["h_pre"], ["h"]),
    helper.make_node("Gemm", ["h", "W2", "b2"], ["engagement"]),
    helper.make_node("Gemm", ["h", "W3", "b3"], ["states"]),
]
graph = helper.make_graph(
    nodes, "dummy_engagement",
    [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 30, 13])],
    [helper.make_tensor_value_info("engagement", TensorProto.FLOAT, [1, 4]),
     helper.make_tensor_value_info("states", TensorProto.FLOAT, [1, 4])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "web/public/model/model_int8.onnx")
print("saved web/public/model/model_int8.onnx")
```

- [ ] **Step 2: Generate and verify locally**

```powershell
pip install onnx onnxruntime numpy
python scripts/make_dummy_onnx.py
python -c "import onnxruntime as rt, numpy as np; s = rt.InferenceSession('web/public/model/model_int8.onnx'); o = s.run(None, {'features': np.random.randn(1,30,13).astype(np.float32)}); print([x.shape for x in o])"
```
Expected: `[(1, 4), (1, 4)]`.

- [ ] **Step 3: Placeholder scaler** — `web/public/model/scaler.json`:

```json
{
  "mean": [0,0,0,0,0,0,0,0,0,0,0,0,0],
  "std": [1,1,1,1,1,1,1,1,1,1,1,1,1],
  "feature_names": ["ear_left","ear_right","ear_mean","mar","brow_left","brow_right","yaw","pitch","roll","gaze_x","gaze_y","face_area","face_present"],
  "pitch_centre": 0.5,
  "version": "1.0"
}
```

- [ ] **Step 4: Commit** — `git add scripts web/public/model && git commit -m "feat: dummy ONNX model with contract I/O + placeholder scaler.json"`

---

### Task 3: Self-host runtime assets (ort WASM, MediaPipe WASM, face_landmarker.task)

**Files:**
- Create: `web/public/ort/` (copied from node_modules)
- Create: `web/public/mediapipe/wasm/` (copied from node_modules)
- Create: `web/public/models/face_landmarker.task` (downloaded)
- Create: `web/scripts/copy-assets.mjs` + `postinstall` script

**Interfaces:**
- Produces: `/ort/` (ort `.wasm`/`.mjs`), `/mediapipe/wasm/` (fileset), `/models/face_landmarker.task` — the exact paths used by Tasks 4 and 6.

- [ ] **Step 1: Copy script** — `web/scripts/copy-assets.mjs`:

```js
import { cpSync, mkdirSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
const root = new URL('..', import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1');
mkdirSync(join(root, 'public/ort'), { recursive: true });
const ortDist = join(root, 'node_modules/onnxruntime-web/dist');
for (const f of readdirSync(ortDist)) {
  if (f.endsWith('.wasm') || f.endsWith('.mjs')) cpSync(join(ortDist, f), join(root, 'public/ort', f));
}
cpSync(join(root, 'node_modules/@mediapipe/tasks-vision/wasm'), join(root, 'public/mediapipe/wasm'), { recursive: true });
console.log('assets copied');
```
Add `"postinstall": "node scripts/copy-assets.mjs"` to `web/package.json` and run it once now.

- [ ] **Step 2: Download the task model** (~3.7 MB, one time, committed):

```powershell
curl -L -o web/public/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

- [ ] **Step 3: Verify** — files exist: `web/public/ort/*.wasm`, `web/public/mediapipe/wasm/vision_wasm_internal.wasm`, `.task` file > 3 MB.

- [ ] **Step 4: Commit** — ensure `web/.gitignore` doesn't exclude `public`; `git add web && git commit -m "feat(web): self-host ort/mediapipe wasm and face_landmarker.task"`

---

### Task 4: features.ts — contract feature extraction with unit tests (B4)

**Files:**
- Create: `web/lib/features.ts`
- Test: `web/tests/features.test.ts`

**Interfaces:**
- Produces: `FEATURE_NAMES: string[]` (13, contract order); `computeFeatures(landmarks: {x,y,z}[] | null | undefined, frameWidth: number, frameHeight: number, pitchCentre: number): Float32Array` (length 13). Consumed by Tasks 5, 7, 8.

- [ ] **Step 1: Write failing tests** — `web/tests/features.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { computeFeatures, FEATURE_NAMES } from '../lib/features';

const W = 100, H = 100, PC = 0.5;

function blankLandmarks(): { x: number; y: number; z: number }[] {
  return Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
}

describe('computeFeatures', () => {
  it('has the frozen 13-name order', () => {
    expect(FEATURE_NAMES).toEqual([
      'ear_left','ear_right','ear_mean','mar','brow_left','brow_right',
      'yaw','pitch','roll','gaze_x','gaze_y','face_area','face_present']);
  });

  it('missing face -> thirteen zeros, face_present 0', () => {
    const f = computeFeatures(null, W, H, PC);
    expect(Array.from(f)).toEqual(new Array(13).fill(0));
  });

  it('468 landmarks (no iris) -> treated as missing face', () => {
    const f = computeFeatures(blankLandmarks().slice(0, 468), W, H, PC);
    expect(f[12]).toBe(0);
  });

  it('computes EAR from the contract formula', () => {
    const lm = blankLandmarks();
    // left eye p1..p6 = [33,160,158,133,153,144]; craft |p2-p6|=|p3-p5|=0.02H, |p1-p4|=0.1W
    lm[33]  = { x: 0.30, y: 0.50, z: 0 }; lm[133] = { x: 0.40, y: 0.50, z: 0 };
    lm[160] = { x: 0.33, y: 0.49, z: 0 }; lm[144] = { x: 0.33, y: 0.51, z: 0 };
    lm[158] = { x: 0.37, y: 0.49, z: 0 }; lm[153] = { x: 0.37, y: 0.51, z: 0 };
    const f = computeFeatures(lm, W, H, PC);
    // EAR = (2 + 2) / (2 * 10) = 0.2
    expect(f[0]).toBeCloseTo(0.2, 5);
    expect(f[12]).toBe(1);
  });

  it('level eyes -> roll 0; symmetric nose -> yaw 0', () => {
    const lm = blankLandmarks();
    lm[33] = { x: 0.3, y: 0.5, z: 0 }; lm[263] = { x: 0.7, y: 0.5, z: 0 };
    lm[1] = { x: 0.5, y: 0.6, z: 0 };
    const f = computeFeatures(lm, W, H, PC);
    expect(f[8]).toBeCloseTo(0, 5);
    expect(f[6]).toBeCloseTo(0, 5);
  });

  it('pitch follows contract formula', () => {
    const lm = blankLandmarks();
    lm[33] = { x: 0.3, y: 0.4, z: 0 }; lm[263] = { x: 0.7, y: 0.4, z: 0 };
    lm[1] = { x: 0.5, y: 0.55, z: 0 }; lm[152] = { x: 0.5, y: 0.7, z: 0 };
    // pitch = (55-40)/|70-40| - 0.5 = 0.5 - 0.5 = 0
    const f = computeFeatures(lm, W, H, PC);
    expect(f[7]).toBeCloseTo(0, 4);
  });
});
```

- [ ] **Step 2: Run** — `npm test` → FAIL (module not found).

- [ ] **Step 3: Implement** — `web/lib/features.ts`:

```ts
// Port target: ml/src/features.py (CONTRACT.md sections 2-4). Conventions:
// PIXEL coords, IMAGE-LEFT naming, un-mirrored frames, y grows downward.
export const FEATURE_NAMES = [
  'ear_left','ear_right','ear_mean','mar','brow_left','brow_right',
  'yaw','pitch','roll','gaze_x','gaze_y','face_area','face_present',
] as const;

const LEFT_EYE_EAR  = [33, 160, 158, 133, 153, 144];   // p1..p6
const RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380];  // p1..p6
const MOUTH         = [61, 291, 13, 14];               // left, right, upper, lower
const LEFT_BROW     = [70, 63, 105, 66, 107];
const RIGHT_BROW    = [300, 293, 334, 296, 336];
const INTEROCULAR   = [33, 263];
const NOSE_TIP = 1;
const CHIN = 152;
const LEFT_IRIS  = [468, 469, 470, 471, 472];
const RIGHT_IRIS = [473, 474, 475, 476, 477];
const EPS = 1e-8;

export interface Landmark { x: number; y: number; z: number }
interface Pt { x: number; y: number }

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const mean = (pts: Pt[]): Pt => ({
  x: pts.reduce((s, p) => s + p.x, 0) / pts.length,
  y: pts.reduce((s, p) => s + p.y, 0) / pts.length,
});

function ear(px: Pt[], idx: number[]): number {
  const [p1, p2, p3, p4, p5, p6] = idx.map((i) => px[i]);
  return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4) + EPS);
}

function gaze(px: Pt[], iris: number[], outer: number, inner: number, earIdx: number[]) {
  const centre = mean(iris.map((i) => px[i]));
  const c1 = px[outer], c2 = px[inner];
  const mid = mean([c1, c2]);
  const width = dist(c1, c2);
  // J1-CHECK: eye height defined as mean of the two vertical EAR distances.
  const [, p2, p3, , p5, p6] = earIdx.map((i) => px[i]);
  const height = (dist(p2, p6) + dist(p3, p5)) / 2;
  return { gx: (centre.x - mid.x) / (width + EPS), gy: (centre.y - mid.y) / (height + EPS) };
}

export function computeFeatures(
  landmarks: Landmark[] | null | undefined,
  frameWidth: number,
  frameHeight: number,
  pitchCentre: number,
): Float32Array {
  const out = new Float32Array(13); // all zeros, face_present = 0
  if (!landmarks || landmarks.length < 478) return out;

  const px: Pt[] = landmarks.map((l) => ({ x: l.x * frameWidth, y: l.y * frameHeight }));

  const leftEyeOuter = px[INTEROCULAR[0]];
  const rightEyeOuter = px[INTEROCULAR[1]];
  const interocular = dist(leftEyeOuter, rightEyeOuter) + EPS;

  const earLeft = ear(px, LEFT_EYE_EAR);
  const earRight = ear(px, RIGHT_EYE_EAR);

  const [mL, mR, mU, mD] = MOUTH.map((i) => px[i]);
  const mar = dist(mU, mD) / (dist(mL, mR) + EPS);

  // J1-CHECK: brow = distance(mean of 5 brow pts, mean of 6 eye pts) / interocular.
  const browLeft = dist(mean(LEFT_BROW.map((i) => px[i])), mean(LEFT_EYE_EAR.map((i) => px[i]))) / interocular;
  const browRight = dist(mean(RIGHT_BROW.map((i) => px[i])), mean(RIGHT_EYE_EAR.map((i) => px[i]))) / interocular;

  const nose = px[NOSE_TIP];
  const chin = px[CHIN];
  const roll = Math.atan2(rightEyeOuter.y - leftEyeOuter.y, rightEyeOuter.x - leftEyeOuter.x);
  const dL = dist(nose, leftEyeOuter);
  const dR = dist(nose, rightEyeOuter);
  const yaw = (dL - dR) / (dL + dR + EPS);
  const eyeMid = mean([leftEyeOuter, rightEyeOuter]);
  const pitch = (nose.y - eyeMid.y) / (Math.abs(chin.y - eyeMid.y) + EPS) - pitchCentre;

  const gL = gaze(px, LEFT_IRIS, 33, 133, LEFT_EYE_EAR);
  const gR = gaze(px, RIGHT_IRIS, 263, 362, RIGHT_EYE_EAR);
  const gazeX = (gL.gx + gR.gx) / 2;
  const gazeY = (gL.gy + gR.gy) / 2;

  // J1-CHECK: face bbox from landmark extremes (pixels) / frame area.
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of px) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  const faceArea = ((maxX - minX) * (maxY - minY)) / (frameWidth * frameHeight);

  out.set([earLeft, earRight, (earLeft + earRight) / 2, mar, browLeft, browRight,
           yaw, pitch, roll, gazeX, gazeY, faceArea, 1.0]);
  return out;
}
```

- [ ] **Step 4: Run** — `npm test` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(web): contract feature extraction (13 features) with unit tests"`

---

### Task 5: Ring buffer + math utils with tests (B5 part 1)

**Files:**
- Create: `web/lib/ringBuffer.ts`, `web/lib/mathUtils.ts`
- Test: `web/tests/ringBuffer.test.ts`, `web/tests/mathUtils.test.ts`

**Interfaces:**
- Produces: `class RingBuffer { push(f: Float32Array): void; isFull(): boolean; count: number; window(): Float32Array /* 390, oldest→newest */ }`; `softmax(x: Float32Array | number[]): number[]`; `sigmoid(x: Float32Array | number[]): number[]`; `standardise(win: Float32Array, mean: number[], std: number[]): Float32Array`.

- [ ] **Step 1: Failing tests**

`web/tests/ringBuffer.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { RingBuffer } from '../lib/ringBuffer';

const frame = (v: number) => new Float32Array(13).fill(v);

describe('RingBuffer', () => {
  it('is not full until 30 frames pushed', () => {
    const rb = new RingBuffer();
    for (let i = 0; i < 29; i++) rb.push(frame(i));
    expect(rb.isFull()).toBe(false);
    rb.push(frame(29));
    expect(rb.isFull()).toBe(true);
  });

  it('window is oldest->newest and evicts', () => {
    const rb = new RingBuffer();
    for (let i = 0; i < 31; i++) rb.push(frame(i)); // frame 0 evicted
    const w = rb.window();
    expect(w.length).toBe(390);
    expect(w[0]).toBe(1);        // oldest remaining
    expect(w[389]).toBe(30);     // newest
  });
});
```

`web/tests/mathUtils.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { softmax, sigmoid, standardise } from '../lib/mathUtils';

describe('mathUtils', () => {
  it('softmax sums to 1 and orders correctly', () => {
    const p = softmax([1, 2, 3, 4]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(p[3]).toBeGreaterThan(p[0]);
  });
  it('sigmoid(0) = 0.5', () => {
    expect(sigmoid([0])[0]).toBeCloseTo(0.5, 6);
  });
  it('standardise applies (x-mean)/std element-wise per feature', () => {
    const win = new Float32Array(390).fill(2);
    const mean = new Array(13).fill(1), std = new Array(13).fill(2);
    const s = standardise(win, mean, std);
    expect(s[0]).toBeCloseTo(0.5, 6);
    expect(s[389]).toBeCloseTo(0.5, 6);
  });
});
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

`web/lib/ringBuffer.ts`:
```ts
const FRAME = 13;
const WINDOW = 30;

export class RingBuffer {
  private buf = new Float32Array(WINDOW * FRAME);
  count = 0;

  push(f: Float32Array): void {
    if (f.length !== FRAME) throw new Error(`expected ${FRAME} features, got ${f.length}`);
    this.buf.copyWithin(0, FRAME);
    this.buf.set(f, (WINDOW - 1) * FRAME);
    this.count++;
  }

  isFull(): boolean {
    return this.count >= WINDOW;
  }

  window(): Float32Array {
    return this.buf;
  }
}
```

`web/lib/mathUtils.ts`:
```ts
export function softmax(x: Float32Array | number[]): number[] {
  const m = Math.max(...Array.from(x));
  const e = Array.from(x, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function sigmoid(x: Float32Array | number[]): number[] {
  return Array.from(x, (v) => 1 / (1 + Math.exp(-v)));
}

export function standardise(win: Float32Array, mean: number[], std: number[]): Float32Array {
  const out = new Float32Array(win.length);
  for (let i = 0; i < win.length; i++) {
    const f = i % 13;
    out[i] = (win[i] - mean[f]) / (std[f] || 1e-8);
  }
  return out;
}
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(web): ring buffer and softmax/sigmoid/standardise with tests"`

---

### Task 6: Scaler loader + inference session (B5 part 2, B1 done-check)

**Files:**
- Create: `web/lib/scaler.ts`, `web/lib/inference.ts`
- Test: `web/tests/scaler.test.ts`

**Interfaces:**
- Consumes: `FEATURE_NAMES` (Task 4), `softmax/sigmoid/standardise` (Task 5), `/model/model_int8.onnx`, `/model/scaler.json` (Task 2), `/ort/` (Task 3).
- Produces: `interface Scaler { mean: number[]; std: number[]; feature_names: string[]; pitch_centre: number; version: string }`; `validateScaler(s: unknown): Scaler` (throws on mismatch); `loadScaler(url?: string): Promise<Scaler>`; `initInference(modelUrl?: string): Promise<{ backend: string; threads: number; modelBytes: number }>`; `runInference(win: Float32Array, scaler: Scaler): Promise<{ engagement: number[]; states: number[]; ms: number }>` (engagement softmaxed, states sigmoided).

- [ ] **Step 1: Failing test** — `web/tests/scaler.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { validateScaler } from '../lib/scaler';
import { FEATURE_NAMES } from '../lib/features';

const good = {
  mean: new Array(13).fill(0), std: new Array(13).fill(1),
  feature_names: [...FEATURE_NAMES], pitch_centre: 0.5, version: '1.0',
};

describe('validateScaler', () => {
  it('accepts the contract schema', () => {
    expect(validateScaler(good).pitch_centre).toBe(0.5);
  });
  it('THROWS on feature_names mismatch', () => {
    const bad = { ...good, feature_names: [...FEATURE_NAMES].reverse() };
    expect(() => validateScaler(bad)).toThrow(/feature_names/);
  });
  it('throws on wrong lengths', () => {
    expect(() => validateScaler({ ...good, mean: [0] })).toThrow();
  });
});
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**

`web/lib/scaler.ts`:
```ts
import { FEATURE_NAMES } from './features';

export interface Scaler {
  mean: number[];
  std: number[];
  feature_names: string[];
  pitch_centre: number;
  version: string;
}

export function validateScaler(s: unknown): Scaler {
  const sc = s as Scaler;
  if (!Array.isArray(sc.mean) || sc.mean.length !== 13) throw new Error('scaler.mean must be 13 floats');
  if (!Array.isArray(sc.std) || sc.std.length !== 13) throw new Error('scaler.std must be 13 floats');
  if (!Array.isArray(sc.feature_names) || sc.feature_names.length !== 13) {
    throw new Error('scaler.feature_names must be 13 strings');
  }
  sc.feature_names.forEach((n, i) => {
    if (n !== FEATURE_NAMES[i]) {
      throw new Error(`scaler feature_names[${i}]="${n}" != contract "${FEATURE_NAMES[i]}" — refusing to run`);
    }
  });
  if (typeof sc.pitch_centre !== 'number') throw new Error('scaler.pitch_centre missing');
  return sc;
}

export async function loadScaler(url = '/model/scaler.json'): Promise<Scaler> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`scaler.json fetch failed: ${res.status}`);
  return validateScaler(await res.json());
}
```

`web/lib/inference.ts`:
```ts
import * as ort from 'onnxruntime-web';
import { softmax, sigmoid, standardise } from './mathUtils';
import type { Scaler } from './scaler';

let session: ort.InferenceSession | null = null;

export async function initInference(modelUrl = '/model/model_int8.onnx') {
  if (session) return sessionInfo;
  ort.env.wasm.wasmPaths = '/ort/';
  ort.env.wasm.numThreads = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency ?? 4 : 4;
  ort.env.wasm.simd = true;
  const res = await fetch(modelUrl);
  if (!res.ok) throw new Error(`model fetch failed: ${res.status}`);
  const bytes = await res.arrayBuffer();
  session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  sessionInfo = { backend: 'wasm', threads: ort.env.wasm.numThreads as number, modelBytes: bytes.byteLength };
  return sessionInfo;
}

let sessionInfo = { backend: 'wasm', threads: 0, modelBytes: 0 };

export async function runInference(win: Float32Array, scaler: Scaler) {
  if (!session) throw new Error('initInference() not called');
  const t0 = performance.now();
  const std = standardise(win, scaler.mean, scaler.std);
  const tensor = new ort.Tensor('float32', std, [1, 30, 13]);
  const out = await session.run({ features: tensor });
  const ms = performance.now() - t0;
  return {
    engagement: softmax(out.engagement.data as Float32Array),
    states: sigmoid(out.states.data as Float32Array),
    ms,
  };
}
```

- [ ] **Step 4: Run** — `npm test` → PASS (scaler tests; inference is browser-verified in Task 8).
- [ ] **Step 5: Commit** — `git commit -m "feat(web): scaler validation (throws on mismatch) and ONNX wasm inference session"`

---

### Task 7: Webcam + MediaPipe landmarker + overlay (B2, B3)

**Files:**
- Create: `web/components/WebcamFeed.tsx`, `web/components/LandmarkOverlay.tsx`, `web/lib/faceLandmarker.ts`

**Interfaces:**
- Consumes: `/mediapipe/wasm/`, `/models/face_landmarker.task` (Task 3).
- Produces: `createLandmarker(): Promise<FaceLandmarker>` (GPU delegate, CPU fallback); `WebcamFeed({ onVideoReady: (v: HTMLVideoElement) => void, mirrored?: boolean })` — renders video, handles denied/no-camera/busy as UI states, stops tracks on unmount; `LandmarkOverlay({ landmarks, videoWidth, videoHeight, mirrored })` — canvas dot overlay.

- [ ] **Step 1: Landmarker lib** — `web/lib/faceLandmarker.ts`:

```ts
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

export async function createLandmarker(): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
  const opts = (delegate: 'GPU' | 'CPU') => ({
    baseOptions: { modelAssetPath: '/models/face_landmarker.task', delegate },
    runningMode: 'VIDEO' as const,
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false, // contract 3.2: never use these
  });
  try {
    return await FaceLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch (e) {
    console.warn('GPU delegate failed, falling back to CPU', e);
    return await FaceLandmarker.createFromOptions(fileset, opts('CPU'));
  }
}
```

- [ ] **Step 2: WebcamFeed** — `web/components/WebcamFeed.tsx`:

```tsx
'use client';
import { useEffect, useRef, useState } from 'react';

type CamState = 'starting' | 'active' | 'denied' | 'nocamera' | 'busy' | 'error';

const MESSAGES: Record<Exclude<CamState, 'active'>, string> = {
  starting: 'Starting camera…',
  denied: 'Camera permission denied. Allow camera access in the browser address bar, then reload.',
  nocamera: 'No camera found on this device.',
  busy: 'Camera is in use by another application. Close it and reload.',
  error: 'Could not start the camera.',
};

export default function WebcamFeed({
  onVideoReady,
  mirrored = true,
}: {
  onVideoReady: (v: HTMLVideoElement) => void;
  mirrored?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [state, setState] = useState<CamState>('starting');

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: 30 },
          audio: false,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        const v = videoRef.current!;
        v.srcObject = stream;
        await v.play();
        setState('active');
        onVideoReady(v);
      } catch (e) {
        const err = e as DOMException;
        if (err.name === 'NotAllowedError') setState('denied');
        else if (err.name === 'NotFoundError') setState('nocamera');
        else if (err.name === 'NotReadableError') setState('busy');
        else setState('error');
      }
    })();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop()); // camera light must turn off
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="relative w-full aspect-[4/3] bg-black rounded-xl overflow-hidden">
      {/* Raw frame goes to the landmarker; mirroring is CSS-only (contract 4.2) */}
      <video
        ref={videoRef}
        playsInline
        muted
        className="h-full w-full object-cover"
        style={mirrored ? { transform: 'scaleX(-1)' } : undefined}
      />
      {state !== 'active' && (
        <div className="absolute inset-0 grid place-items-center bg-zinc-950/90 p-6 text-center text-zinc-300">
          {MESSAGES[state]}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Overlay** — `web/components/LandmarkOverlay.tsx`:

```tsx
'use client';
import { useEffect, useRef } from 'react';
import type { Landmark } from '../lib/features';

export default function LandmarkOverlay({
  landmarks, mirrored = true,
}: { landmarks: Landmark[] | null; mirrored?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    if (!landmarks) return;
    ctx.fillStyle = '#22d3ee';
    for (const l of landmarks) {
      const x = (mirrored ? 1 - l.x : l.x) * c.width;
      ctx.fillRect(x - 0.5, l.y * c.height - 0.5, 1.5, 1.5);
    }
  }, [landmarks, mirrored]);
  return <canvas ref={ref} width={640} height={480}
    className="pointer-events-none absolute inset-0 h-full w-full" />;
}
```

- [ ] **Step 4: Commit** — `git commit -m "feat(web): webcam feed with failure states, mediapipe landmarker, overlay"`

---

### Task 8: Pipeline hook + main page wiring (B5 done, app works end-to-end)

**Files:**
- Create: `web/hooks/usePipeline.ts`
- Modify: `web/app/page.tsx` (replace boilerplate)

**Interfaces:**
- Consumes: everything above.
- Produces: `usePipeline()` returning `{ status: string; error: string | null; landmarks: Landmark[] | null; features: Float32Array | null; prediction: { engagement: number[]; states: number[]; ms: number } | null; perf: { renderFps: number; sampleHz: number; inferMs: number[]; modelBytes: number; backend: string; threads: number; landmarkCount: number }; onVideoReady: (v: HTMLVideoElement) => void }`. Timing: rAF loop counts render FPS; samples features when `now - lastSample >= 100` ms; infers when `buffer.isFull() && sampleCount % 5 === 0`, guarded by an in-flight flag.

- [ ] **Step 1: Implement** — `web/hooks/usePipeline.ts`:

```ts
'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { createLandmarker } from '../lib/faceLandmarker';
import { computeFeatures, type Landmark } from '../lib/features';
import { RingBuffer } from '../lib/ringBuffer';
import { loadScaler, type Scaler } from '../lib/scaler';
import { initInference, runInference } from '../lib/inference';

export interface Prediction { engagement: number[]; states: number[]; ms: number }

export function usePipeline() {
  const [status, setStatus] = useState('loading models…');
  const [error, setError] = useState<string | null>(null);
  const [landmarks, setLandmarks] = useState<Landmark[] | null>(null);
  const [features, setFeatures] = useState<Float32Array | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [perf, setPerf] = useState({
    renderFps: 0, sampleHz: 0, inferMs: [] as number[],
    modelBytes: 0, backend: '-', threads: 0, landmarkCount: 0,
  });

  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const scalerRef = useRef<Scaler | null>(null);
  const bufferRef = useRef(new RingBuffer());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const rafRef = useRef(0);
  const lastSampleRef = useRef(0);
  const sampleCountRef = useRef(0);
  const inFlightRef = useRef(false);
  const fpsCounter = useRef({ frames: 0, samples: 0, last: performance.now() });
  const inferTimes = useRef<number[]>([]);

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const [lmk, scaler, info] = await Promise.all([
          createLandmarker(), loadScaler(), initInference(),
        ]);
        if (dead) return;
        landmarkerRef.current = lmk;
        scalerRef.current = scaler;
        setPerf((p) => ({ ...p, modelBytes: info.modelBytes, backend: info.backend, threads: info.threads }));
        setStatus('waiting for camera');
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { dead = true; cancelAnimationFrame(rafRef.current); };
  }, []);

  const loop = useCallback((now: number) => {
    rafRef.current = requestAnimationFrame(loop);
    const video = videoRef.current, lmk = landmarkerRef.current, scaler = scalerRef.current;
    if (!video || !lmk || !scaler || video.readyState < 2) return;

    const c = fpsCounter.current;
    c.frames++;
    if (now - c.last >= 1000) {
      setPerf((p) => ({ ...p, renderFps: c.frames, sampleHz: c.samples, inferMs: [...inferTimes.current] }));
      c.frames = 0; c.samples = 0; c.last = now;
    }

    if (now - lastSampleRef.current < 100) return; // contract: 10 Hz sampling
    lastSampleRef.current = now;
    c.samples++;

    const result = lmk.detectForVideo(video, now);
    const lm = (result.faceLandmarks[0] as Landmark[] | undefined) ?? null;
    if (lm && lm.length !== 478) console.error(`landmark count ${lm.length}, expected 478 — iris missing?`);
    setLandmarks(lm);
    setPerf((p) => (p.landmarkCount === (lm?.length ?? 0) ? p : { ...p, landmarkCount: lm?.length ?? 0 }));

    const f = computeFeatures(lm, video.videoWidth, video.videoHeight, scaler.pitch_centre);
    setFeatures(f);
    bufferRef.current.push(f);
    sampleCountRef.current++;

    if (bufferRef.current.isFull() && sampleCountRef.current % 5 === 0 && !inFlightRef.current) {
      inFlightRef.current = true;
      runInference(bufferRef.current.window(), scaler)
        .then((pred) => {
          inferTimes.current = [...inferTimes.current.slice(-29), pred.ms];
          setPrediction(pred);
          setStatus('live');
        })
        .catch((e) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => { inFlightRef.current = false; });
    }
  }, []);

  const onVideoReady = useCallback((v: HTMLVideoElement) => {
    videoRef.current = v;
    setStatus('filling 3s window…');
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(loop);
  }, [loop]);

  return { status, error, landmarks, features, prediction, perf, onVideoReady };
}
```

- [ ] **Step 2: Minimal page proving E2E** — replace `web/app/page.tsx` (full UI lands in Task 9; this step must show camera + overlay + raw prediction JSON):

```tsx
'use client';
import WebcamFeed from '../components/WebcamFeed';
import LandmarkOverlay from '../components/LandmarkOverlay';
import { usePipeline } from '../hooks/usePipeline';

export default function Home() {
  const { status, error, landmarks, prediction, perf, onVideoReady } = usePipeline();
  return (
    <main className="min-h-screen bg-zinc-950 p-6 text-zinc-100">
      <div className="relative max-w-xl">
        <WebcamFeed onVideoReady={onVideoReady} />
        <LandmarkOverlay landmarks={landmarks} />
      </div>
      <pre className="mt-4 text-xs">{error ?? status}{'\n'}{JSON.stringify({ perf, prediction }, null, 2)}</pre>
    </main>
  );
}
```

- [ ] **Step 3: Verify in browser** — `npm run dev`, open localhost:3000, allow camera. Expected: video + cyan landmark dots tracking the face; after ~3 s predictions update ~2×/s; `landmarkCount: 478`; renderFps ≥ 30; sampleHz ≈ 10. Console has no red errors.
- [ ] **Step 4: Commit** — `git commit -m "feat(web): end-to-end pipeline camera->landmarks->features->buffer->dummy inference"`

---

### Task 9: Dashboard UI — PredictionPanel, FeaturePanel, PerfHUD (B6)

**Files:**
- Create: `web/components/PredictionPanel.tsx`, `web/components/PerfHUD.tsx`, `web/components/FeaturePanel.tsx`, `web/components/Sparkline.tsx`
- Modify: `web/app/page.tsx`, `web/app/layout.tsx` (title/meta)

**Interfaces:**
- Consumes: `usePipeline()` outputs (Task 8), `FEATURE_NAMES` (Task 4).
- Produces: projector-readable dark dashboard, cyan accent (`#22d3ee`), large numerals.

- [ ] **Step 1: Sparkline** — `web/components/Sparkline.tsx` (rolling 60 s of engagement level, canvas):

```tsx
'use client';
import { useEffect, useRef } from 'react';

export default function Sparkline({ values, max = 3 }: { values: number[]; max?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    ctx.clearRect(0, 0, c.width, c.height);
    if (values.length < 2) return;
    ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2; ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * c.width;
      const y = c.height - 4 - (v / max) * (c.height - 8);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [values, max]);
  return <canvas ref={ref} width={560} height={80} className="w-full" />;
}
```

- [ ] **Step 2: PredictionPanel** — `web/components/PredictionPanel.tsx`:

```tsx
'use client';
import Sparkline from './Sparkline';
import type { Prediction } from '../hooks/usePipeline';

const LEVELS = ['Very Low', 'Low', 'High', 'Very High'];
const STATES = ['Bored', 'Confused', 'Engaged', 'Frustrated'];

export default function PredictionPanel({
  prediction, history,
}: { prediction: Prediction | null; history: number[] }) {
  const level = prediction ? prediction.engagement.indexOf(Math.max(...prediction.engagement)) : null;
  return (
    <section className="rounded-2xl bg-zinc-900 p-6">
      <h2 className="text-sm font-semibold uppercase tracking-widest text-zinc-500">Engagement</h2>
      <div className="mt-2 flex items-baseline gap-4">
        <span className="text-6xl font-bold text-cyan-400">{level === null ? '—' : level}</span>
        <span className="text-2xl text-zinc-300">{level === null ? 'warming up' : LEVELS[level]}</span>
      </div>
      <div className="mt-6 space-y-3">
        {STATES.map((name, i) => {
          const p = prediction?.states[i] ?? 0;
          return (
            <div key={name} className="flex items-center gap-3">
              <span className="w-28 text-lg text-zinc-300">{name}</span>
              <div className="h-4 flex-1 rounded bg-zinc-800">
                <div className="h-4 rounded bg-cyan-400 transition-[width] duration-300"
                     style={{ width: `${(p * 100).toFixed(1)}%` }} />
              </div>
              <span className="w-16 text-right text-lg tabular-nums">{(p * 100).toFixed(0)}%</span>
            </div>
          );
        })}
      </div>
      <div className="mt-6">
        <h3 className="text-xs uppercase tracking-widest text-zinc-500">last 60 s</h3>
        <Sparkline values={history} />
      </div>
    </section>
  );
}
```

- [ ] **Step 3: PerfHUD** — `web/components/PerfHUD.tsx`:

```tsx
'use client';

function Stat({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-widest text-zinc-500">{label}</div>
      <div className="text-3xl font-bold tabular-nums text-zinc-100">
        {value}<span className="ml-1 text-base text-zinc-400">{unit}</span>
      </div>
    </div>
  );
}

export default function PerfHUD({ perf }: {
  perf: { renderFps: number; sampleHz: number; inferMs: number[]; modelBytes: number; backend: string; threads: number; landmarkCount: number };
}) {
  const meanMs = perf.inferMs.length
    ? perf.inferMs.reduce((a, b) => a + b, 0) / perf.inferMs.length : 0;
  return (
    <section className="grid grid-cols-3 gap-6 rounded-2xl bg-zinc-900 p-6 md:grid-cols-6">
      <Stat label="Render" value={String(perf.renderFps)} unit="fps" />
      <Stat label="Sampling" value={String(perf.sampleHz)} unit="Hz" />
      <Stat label="Inference" value={meanMs.toFixed(1)} unit="ms" />
      <Stat label="Model" value={(perf.modelBytes / 1024).toFixed(0)} unit="KB" />
      <Stat label="Backend" value={`${perf.backend}×${perf.threads}`} />
      <Stat label="Landmarks" value={String(perf.landmarkCount)} />
    </section>
  );
}
```

- [ ] **Step 4: FeaturePanel** — `web/components/FeaturePanel.tsx` (live 13 values — "tell me all data"):

```tsx
'use client';
import { FEATURE_NAMES } from '../lib/features';

export default function FeaturePanel({ features }: { features: Float32Array | null }) {
  return (
    <section className="rounded-2xl bg-zinc-900 p-6">
      <h2 className="text-sm font-semibold uppercase tracking-widest text-zinc-500">Live features (raw, 10 Hz)</h2>
      <div className="mt-3 grid grid-cols-2 gap-x-8 gap-y-1 font-mono text-sm">
        {FEATURE_NAMES.map((name, i) => (
          <div key={name} className="flex justify-between border-b border-zinc-800 py-1">
            <span className="text-zinc-400">{name}</span>
            <span className="tabular-nums text-zinc-100">{features ? features[i].toFixed(4) : '—'}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 5: Final page layout** — `web/app/page.tsx`:

```tsx
'use client';
import { useEffect, useRef, useState } from 'react';
import WebcamFeed from '../components/WebcamFeed';
import LandmarkOverlay from '../components/LandmarkOverlay';
import PredictionPanel from '../components/PredictionPanel';
import PerfHUD from '../components/PerfHUD';
import FeaturePanel from '../components/FeaturePanel';
import BenchmarkPanel from '../components/BenchmarkPanel';
import { usePipeline } from '../hooks/usePipeline';

export default function Home() {
  const { status, error, landmarks, features, prediction, perf, onVideoReady } = usePipeline();
  const [history, setHistory] = useState<number[]>([]);
  const lastPred = useRef<typeof prediction>(null);

  useEffect(() => {
    if (!prediction || prediction === lastPred.current) return;
    lastPred.current = prediction;
    const level = prediction.engagement.indexOf(Math.max(...prediction.engagement));
    setHistory((h) => [...h.slice(-119), level]); // 120 points @ 2 Hz = 60 s
  }, [prediction]);

  return (
    <main className="min-h-screen bg-zinc-950 p-6 text-zinc-100">
      <header className="mb-6 flex items-baseline justify-between">
        <h1 className="text-2xl font-bold">Cognitive State — In-Browser Inference</h1>
        <span className={`rounded-full px-3 py-1 text-sm ${error ? 'bg-red-900 text-red-200' : 'bg-zinc-800 text-cyan-400'}`}>
          {error ?? status}
        </span>
      </header>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <div className="relative">
            <WebcamFeed onVideoReady={onVideoReady} />
            <LandmarkOverlay landmarks={landmarks} />
          </div>
          <FeaturePanel features={features} />
        </div>
        <div className="space-y-6">
          <PredictionPanel prediction={prediction} history={history} />
          <PerfHUD perf={perf} />
          <BenchmarkPanel />
        </div>
      </div>
    </main>
  );
}
```

(If Task 10 isn't done yet when building this, stub `BenchmarkPanel` as `export default function BenchmarkPanel() { return null; }` — replaced in Task 10.)

Also set metadata in `web/app/layout.tsx`: title `Cognitive Overload Detection — Edge Inference`.

- [ ] **Step 6: Verify in browser** — everything from Task 8 Step 3 plus: bars animate, sparkline draws, no flicker, features tick at 10 Hz.
- [ ] **Step 7: Commit** — `git commit -m "feat(web): dashboard UI - prediction panel, feature panel, perf HUD"`

---

### Task 10: Benchmark harness (B7)

**Files:**
- Create: `web/lib/benchmark.ts`, `web/components/BenchmarkPanel.tsx`
- Create: `docs/benchmarks/.gitkeep`

**Interfaces:**
- Consumes: `runInference`, `initInference` (Task 6), `loadScaler` (Task 6).
- Produces: `runBenchmark(cycles?: number): Promise<BenchmarkResult>` where `BenchmarkResult = { cycles: number; meanMs: number; p50: number; p95: number; p99: number; meanFps: number; heapDeltaMB: number | null; backend: string; threads: number; userAgent: string; timestamp: string }`; a UI button that runs it and downloads `benchmark-<timestamp>.json` (user saves it to `docs/benchmarks/`).

- [ ] **Step 1: Implement** — `web/lib/benchmark.ts`:

```ts
import { initInference, runInference } from './inference';
import { loadScaler } from './scaler';

export interface BenchmarkResult {
  cycles: number; meanMs: number; p50: number; p95: number; p99: number;
  meanFps: number; heapDeltaMB: number | null;
  backend: string; threads: number; userAgent: string; timestamp: string;
}

const pct = (sorted: number[], p: number) =>
  sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];

export async function runBenchmark(cycles = 300): Promise<BenchmarkResult> {
  const info = await initInference();
  const scaler = await loadScaler();
  const win = new Float32Array(390).map(() => Math.random() * 2 - 1);

  const mem = (performance as any).memory;
  const heapBefore = mem?.usedJSHeapSize ?? null;

  for (let i = 0; i < 10; i++) await runInference(win, scaler); // warm-up

  const times: number[] = [];
  for (let i = 0; i < cycles; i++) times.push((await runInference(win, scaler)).ms);

  const heapAfter = mem?.usedJSHeapSize ?? null;
  const sorted = [...times].sort((a, b) => a - b);
  const meanMs = times.reduce((a, b) => a + b, 0) / times.length;
  return {
    cycles, meanMs, p50: pct(sorted, 50), p95: pct(sorted, 95), p99: pct(sorted, 99),
    meanFps: 1000 / meanMs,
    heapDeltaMB: heapBefore != null && heapAfter != null ? (heapAfter - heapBefore) / 1048576 : null,
    backend: info.backend, threads: info.threads,
    userAgent: navigator.userAgent, timestamp: new Date().toISOString(),
  };
}
```

- [ ] **Step 2: Panel** — `web/components/BenchmarkPanel.tsx`:

```tsx
'use client';
import { useState } from 'react';
import { runBenchmark, type BenchmarkResult } from '../lib/benchmark';

export default function BenchmarkPanel() {
  const [result, setResult] = useState<BenchmarkResult | null>(null);
  const [running, setRunning] = useState(false);

  async function go() {
    setRunning(true);
    try {
      const r = await runBenchmark(300);
      setResult(r);
      const blob = new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `benchmark-${r.timestamp.replace(/[:.]/g, '-')}.json`;
      a.click();
      URL.revokeObjectURL(a.href);
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="rounded-2xl bg-zinc-900 p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-zinc-500">Benchmark</h2>
        <button onClick={go} disabled={running}
          className="rounded-lg bg-cyan-500 px-4 py-2 font-semibold text-zinc-950 disabled:opacity-50">
          {running ? 'Running 300 cycles…' : 'Run 300 inferences'}
        </button>
      </div>
      {result && (
        <pre className="mt-4 text-sm text-zinc-300">
{`mean ${result.meanMs.toFixed(2)} ms   p50 ${result.p50.toFixed(2)}   p95 ${result.p95.toFixed(2)}   p99 ${result.p99.toFixed(2)}
${result.meanFps.toFixed(0)} inferences/s   heap Δ ${result.heapDeltaMB?.toFixed(2) ?? 'n/a'} MB`}
        </pre>
      )}
    </section>
  );
}
```

- [ ] **Step 3: Verify** — click the button 3×; numbers stable run-to-run; JSON downloads. Save one JSON into `docs/benchmarks/`.
- [ ] **Step 4: Commit** — `git commit -m "feat(web): benchmark harness - 300 cycles, p50/p95/p99, heap delta, JSON export"`

---

### Task 11: Final verification + docs

**Files:**
- Modify: `README.md` (add web quickstart section)
- Create: `docs/superpowers/plans/…` checkboxes updated

- [ ] **Step 1: Full test suite** — `cd web && npm test` → all green.
- [ ] **Step 2: Production build** — `npm run build` → succeeds; `npm start` → app works same as dev (headers apply in prod too).
- [ ] **Step 3: Manual robustness pass** — cover camera (features → zeros, `face_present 0`, no crash); look away; close eyes (EAR drops in FeaturePanel); reload with camera blocked (clean denied message).
- [ ] **Step 4: README quickstart**:

```markdown
## Web app (Track B)

    cd web
    npm install        # also copies wasm assets via postinstall
    npm run dev        # http://localhost:3000, allow camera

Currently running against the DUMMY model (`scripts/make_dummy_onnx.py`).
Swap `web/public/model/model_int8.onnx` + `scaler.json` for Bilal's real
files on Day 11 — nothing else changes (B8).
```

- [ ] **Step 5: Commit** — `git commit -m "docs: web quickstart + plan checkboxes"`

---

## Out of scope today (needs Bilal's deliverables)

- **J1 parity test** (needs `parity_clip.mp4` + `parity_expected.json`) — `features.ts` carries `J1-CHECK` markers at every interpretation point.
- **B8 real model swap**, **fp32 vs int8 comparison** (Day 12+).
- **B9 privacy proof / CSP `connect-src 'none'`** (Day 18) — note: self-hosting everything today already makes this trivial later.
- Three-machine benchmarks, cross-browser matrix (Days 15/17).

## Self-review notes

- Spec coverage: B1 (Tasks 1–3), B2 (Task 7), B3 (Tasks 3, 7, 8), B4 (Task 4, contract-derived pending Bilal's features.py), B5 (Tasks 5, 6, 8), B6 (Task 9), B7 (Task 10), done-criteria (Task 11). ✔
- Failure-mode table addressed: COOP/COEP (Task 1), 478 assert (Task 8), session-once (Task 6), `isFull` guard (Task 8), CPU fallback (Task 7), self-hosted assets (Task 3), ring-buffer eviction (Task 5). ✔
- Type consistency: `Landmark`, `Scaler`, `Prediction`, `BenchmarkResult`, `perf` shape used identically across Tasks 4–10. ✔
