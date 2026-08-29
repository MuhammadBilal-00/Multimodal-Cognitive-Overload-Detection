# Track B — in-browser inference app

Next.js (App Router, TypeScript strict) application that runs the entire
cognitive-state pipeline client-side: webcam → MediaPipe FaceLandmarker
(WASM, CPU delegate) → 13 geometric features (`lib/features.ts`, a literal
port of `ml/src/features.py` per `../CONTRACT.md` §2–4) → 30-frame ring
buffer → standardise via `public/model/scaler.json` → ONNX Runtime Web
(int8, `public/model/model_int8.onnx`) → dashboard. No frame, feature, or
prediction ever leaves the machine; `next.config.mjs` enforces the egress
lock and cross-origin isolation (see `../docs/privacy.md`).

## Setup

Requires Node.js 20+.

```powershell
npm install    # postinstall copies onnxruntime-web + MediaPipe WASM into
               # public/ort and public/mediapipe (gitignored, regenerated)
```

Download the MediaPipe model asset (gitignored — self-hosted so the app
works offline; same file Track A uses):

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile public\models\face_landmarker.task
```

```powershell
npm run dev    # http://localhost:3000 — allow camera access
npm run build && npm start   # production
```

## Tests

```powershell
npm test              # vitest unit suite (features, states order, scaler,
                      # ring buffer, primary face, inference contract)
npm run test:parity   # J1 Python<->TS feature-parity gate (Playwright;
                      # needs the DAiSEE-derived fixture — see
                      # ml/scripts/make_parity_fixture.py — and the model
                      # asset above; writes docs/results/parity_report.json)
```

## Load-bearing invariants (each has a guard; do not relax casually)

- **Two separate landmarkers** (`lib/faceLandmarker.ts`): `numFaces: 1`
  feeds the model (matches training extraction; `numFaces: 4` shifts
  landmarks enough to fail J1 on blinks — CONTRACT.md Amendment 2);
  `numFaces: 4` drives the overlay/People count only.
- **CPU delegate only** — the GPU delegate fails the parity gate
  (`docs/results/parity_report_gpu.json`).
- **States channel order** is `lib/states.ts` (`boredom, engagement,
  confusion, frustration`), guarded by `tests/states.test.ts`, which parses
  `ml/src/labels.py` from disk — CONTRACT.md §5 Amendment 3.
- **Brow eye-centre is the corner midpoint**, not the 6-landmark centroid
  — CONTRACT.md Amendment 4; guarded by `tests/features.test.ts`.
- Softmax on `engagement`, sigmoid on `states`, standardisation outside
  the graph — CONTRACT.md §5; guarded by `tests/inferenceContract.test.ts`.

`/parity-test` and `/api/parity-fixtures` are development/test-only routes
(disabled in production builds) used by the J1 gate.
