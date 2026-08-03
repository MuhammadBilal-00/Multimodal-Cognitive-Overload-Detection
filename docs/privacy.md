# Privacy proof — Track B

**Claim:** no video frame, image, or extracted feature ever leaves the
device. All inference runs in the browser.

This document is evidence, not assertion: what was tested, how, and one
real finding that changed the shipped code.

---

## 1. Code walkthrough — where a leak could happen, and why it doesn't

Three places touch camera data. Each is traced to confirm nothing it
handles is ever serialized or sent anywhere:

- **`components/WebcamFeed.tsx`** — `getUserMedia` gives a `MediaStream`,
  assigned straight to a `<video>` element's `srcObject`. The stream
  itself never leaves this component; on unmount every track is
  `.stop()`'d (the camera indicator light actually turns off).
- **`lib/faceLandmarker.ts`** / **`hooks/usePipeline.ts`** — the `<video>`
  element is passed directly to `landmarker.detectForVideo(video, now)`.
  No `ImageData`, canvas snapshot, or pixel buffer is ever extracted from
  the frame; MediaPipe reads the video element in-place inside its own
  WASM memory.
- **`lib/features.ts`** → **`lib/ringBuffer.ts`** → **`lib/inference.ts`**
  — the pipeline turns landmarks into a 13-float feature vector, buffers
  30 of them (a plain `Float32Array`, function/hook-local — never
  assigned to a global, never passed to `JSON.stringify`, never touched
  by any network API), and hands that buffer directly to a **local**
  ONNX Runtime Web session. The prediction stays in React state, rendered
  to the DOM.

Grepping the entire `web/` app source (excluding `node_modules`) for every
network-capable API turns up exactly two calls, both to hardcoded,
relative, same-origin paths:

```
web/lib/scaler.ts:28:    const res = await fetch(url);       // default: /model/scaler.json
web/lib/inference.ts:28: const res = await fetch(modelUrl);  // default: /model/model_int8.onnx
```

No `XMLHttpRequest`, `WebSocket`, `EventSource`, `sendBeacon`, or
third-party SDK call exists anywhere in the app's own source. No
CDN/external URL is hardcoded anywhere — `@mediapipe/tasks-vision`'s WASM,
onnxruntime-web's WASM, and the `face_landmarker.task` model asset are all
self-hosted under `/ort/`, `/mediapipe/`, and `/models/` (Task 3 of the
build), specifically so the app has no reason to ever reach an external
host once loaded.

## 2. Network recording — a full live session

Recorded every request Chromium made (Playwright, synthetic camera feed)
from page load through 20–30 seconds of steady-state "live" inference:

| | |
|---|---|
| Distinct origins contacted | 1 — the app's own origin |
| Requests during initial load (page bundle, WASM, model, scaler) | 24 |
| Requests after reaching "live" (camera + landmarks + inference actively running) | **0** |

Every one of the 24 load-time requests is a `GET` to a relative,
same-origin path — the Next.js JS bundle, `/model/scaler.json`,
`/model/model_int8.onnx`, `/ort/*.wasm` / `.mjs`, `/mediapipe/wasm/*`,
`/models/face_landmarker.task`. None repeat once the app is live; the
whole 3-second-window → feature → inference cycle runs with zero
additional network activity for as long as the tab stays open.

## 3. A real finding — MediaPipe's built-in telemetry, and the fix

Before drawing any conclusions from step 2, the same recording was left
running for a full 60+ seconds to make sure nothing was going out on a
slower cadence than a short test would catch. It caught something:

```
POST https://odml.pa.googleapis.com/v1/log
Content-Type: application/x-protobuf
```

This is not from this project's code — it's a usage-telemetry call
baked into `@mediapipe/tasks-vision` (v1.0.1) itself, firing on a 60-second
interval once a `FaceLandmarker` (or any MediaPipe Task) is created. The
payload is a small structured protobuf whose readable fields are just the
library version string (`"1.0.1"`) and a handful of small integers — not
frame data, not landmarks, not anything the size of an image would
require. It appears to be Google's own internal library-usage analytics
(which Task type got created, which version), not a data exfiltration
path.

That distinction doesn't matter for the claim being made here. "Zero
bytes leave the machine" was false as shipped, regardless of what those
bytes contained — and the library exposes no opt-out: `BaseOptions`, the
only public config surface `FaceLandmarker.createFromOptions` accepts,
has exactly three fields (`modelAssetPath`, `modelAssetBuffer`,
`delegate`) and nothing telemetry-related.

**Fix — `next.config.mjs`:**

```js
{ key: 'Content-Security-Policy', value: "connect-src 'self'" }
```

This doesn't patch around the specific telemetry call — it makes *any*
cross-origin `fetch`/`XHR`/`WebSocket`/`sendBeacon`, from this code or any
current or future dependency, fail closed at the browser level. The claim
becomes something the browser enforces, not something every dependency
has to be individually audited and trusted to honor.

**Verified, not assumed:** with the header active, a session was recorded
for 65 seconds (past the logger's 60-second flush interval) with these
results:

| | |
|---|---|
| Requests that actually reached `googleapis.com` | **0** |
| CSP violation reports logged (the blocked attempt) | 2 |
| App-level errors caused by the block | **0** |
| Pipeline still reaches "live" with real predictions | yes |

The two CSP violation lines, verbatim:

```
Connecting to 'https://odml.pa.googleapis.com/v1/log' violates the following
Content Security Policy directive: "connect-src 'self'". The action has been
blocked.
Fetch API cannot load https://odml.pa.googleapis.com/v1/log. Refused to
connect because it violates the document's Content Security Policy.
```

The final network log with the header active (30 seconds of live
inference, well past initial load):

```
distinct origins: [ 'http://localhost:3001' ]
total requests: 24 (identical set to step 2 — all same-origin load-time assets)
requests after reaching "live": 0
```

## 4. What this does and doesn't prove

- **Proves:** with the app as shipped, no code path — first-party or
  third-party — can reach any host except the one serving the page, and
  this holds for a real inference session, not just a static analysis.
- **Doesn't prove:** what a *future* dependency bump might try to do.
  `connect-src 'self'` is exactly the guard for that case: it doesn't rely
  on re-auditing every future `npm update`.
- **Scope:** this is a same-origin deployment (`localhost` today,
  presumably the same single origin in any future deployment). If the app
  is ever split across multiple first-party subdomains, this policy would
  need `connect-src` to list those explicitly rather than relying on
  `'self'`.
