# Privacy-Preserving Cognitive State Detection in the Browser

Final-year project: detecting learner engagement and cognitive states from
webcam video, with **all inference running client-side** — video never leaves
the user's machine. A small temporal convolutional network is trained on the
DAiSEE dataset, quantized to int8 ONNX (~200 KB), and executed in the browser
with onnxruntime-web on features extracted by MediaPipe Face Mesh.

- **Track A** (this repo's `ml/`): DAiSEE video → features → trained,
  quantized ONNX model — Bilal
- **Track B** (`web/`): Next.js app, in-browser feature extraction and
  inference, benchmarking — Azeem

**`CONTRACT.md` is the interface contract between the two tracks. Read it
first.**

## Architecture (Track B — everything below runs in the browser)

```mermaid
flowchart TD
    subgraph Browser["Browser — single origin, self-hosted assets, nothing leaves"]
        Cam["Webcam\ngetUserMedia"] -->|"raw, un-mirrored\nvideo element"| DetF
        Cam -->|"raw, un-mirrored\nvideo element"| DetD

        subgraph Loop["rAF loop — display 30+ fps, sampled 10 Hz"]
            DetF["FaceLandmarker (numFaces:1)\n478 landmarks (incl. iris), WASM\nfeeds the model"]
            DetD["FaceLandmarker (numFaces:4)\ndisplay only — overlay + People count"]
            Feat["features.ts\n13 floats, CONTRACT.md §2-4"]
            Buf["RingBuffer\n30 frames = 3.0 s window"]
            DetF -->|"10 Hz"| Feat --> Buf
        end

        Buf -->|"isFull(), every 30th sample = every 3 s"| Std["standardise()\n(x-mean)/std via scaler.json"]
        Std --> Ort["onnxruntime-web session\nWASM, created once"]
        Ort -->|"engagement[4], states[4]\nraw logits"| Post["softmax / sigmoid\n(in JS, not the graph)"]
        Post --> UI["Dashboard\nPredictionPanel · FeaturePanel · PerfHUD"]
    end

    Assets["Self-hosted:\n/ort/*.wasm · /mediapipe/wasm/*\n/models/face_landmarker.task\n/model/model_int8.onnx + scaler.json"] -.->|"same-origin fetch,\nload-time only"| Loop
    Assets -.-> Ort

    CSP["CSP: connect-src 'self'\n+ COOP/COEP"] -.->|enforces| Browser
```

No server, no CDN, no third-party origin: `next.config.mjs` ships
`Content-Security-Policy: connect-src 'self'`, which makes any cross-origin
network call fail closed at the browser level regardless of what any
dependency tries to do — see `docs/privacy.md` for why that header exists
and the real telemetry call it blocks. Stage-by-stage breakdown (file
references, timing, and the two production-build fixes that shaped this
design) in `docs/architecture.md`.

## Repository layout

```
CONTRACT.md            interface contract (source of truth)
ml/
  requirements.txt     exact pinned Python dependencies
  src/                 feature extraction, dataset, model, training
  tests/               unit tests + parity fixtures
data/                  DAiSEE dataset (gitignored — never committed)
artifacts/             extracted features, training runs (gitignored)
docs/results/          every figure and table destined for the thesis
web/public/model/      shipped model_int8.onnx + scaler.json
```

## Setup (Track A)

Requires Python 3.11 on Windows/Linux/macOS.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r ml\requirements.txt
```

Note: `torch` is pinned as the CPU build; if installing on a fresh machine,
run `pip install torch --index-url https://download.pytorch.org/whl/cpu`
first if the plain requirements install pulls the wrong wheel.

Then download the MediaPipe face-landmark model asset (not committed):

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile ml\assets\face_landmarker.task
```

mediapipe is pinned at **1.0.0**, which has removed the legacy
`mp.solutions.face_mesh` API — all code uses the Tasks API
(`FaceLandmarker`). See CONTRACT.md §4 for why this also helps
Python↔JS parity.

## Setup (Track B)

Requires Node.js 20+.

```powershell
cd web
npm install    # postinstall also runs `npm run assets` (copies onnxruntime-web
               # and @mediapipe/tasks-vision WASM into public/ort and
               # public/mediapipe — gitignored, regenerated every install)
```

Then download the same MediaPipe model asset used by Track A (also not
committed — same file, self-hosted so the demo works offline):

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -OutFile web\public\models\face_landmarker.task
```

```powershell
npm run dev    # http://localhost:3000 — allow camera access
```

The app ships the **real trained** `model_int8.onnx` + `scaler.json` in
`web/public/model/` (since the 2026-08-03 repository merge). The dummy-model
generator (`scripts/make_dummy_onnx.py`, same I/O shapes as CONTRACT.md §5)
remains available for development without the trained weights.

## Dataset

DAiSEE is **not redistributed** in this repository. Request access via the
form at <https://people.iith.ac.in/vineethnb/resources/daisee/index.html>
and extract it to `data/DAiSEE/` preserving the official
Train / Validation / Test folder structure. Budget ~60–80 GB free disk for
the archive plus extracted frames.

### Citation

> A. Gupta, A. D'Cunha, K. Awasthi, V. Balasubramanian.
> *DAiSEE: Towards User Engagement Recognition in the Wild.*
> arXiv:1609.01885, 2016.

> A. Kamath, A. Biswas, V. Balasubramanian.
> *A Crowdsourced Approach to Student Engagement Recognition in e-Learning
> Environments.* IEEE WACV, 2016.

## Status

- [x] Day 1 — repo, contract, environment
- [x] DAiSEE access form submitted, download started (2026-08-01)
- [x] DAiSEE extracted to `data/DAiSEE/` with official Train/Validation/Test folders (5481/1720/1866 clips on disk; labels cover 5358/1429/1784)
- [x] Audio-stream check: **no audio in DAiSEE** — "multimodal" = three visual modality families (CONTRACT.md §8)
- [x] Day 2 — `features.py` + tests (17 passing) + landmark indices visually verified (`docs/verification/`)
- [x] Days 3–5 — extraction (9,032 clips, 0 failures, 99.96% detection), windows + `scaler.json` (pitch_centre 0.3733), parity fixture
- [x] Days 6–7 — TCN (41.5k params), A6.5 browser smoke PASS (static QDQ int8; dynamic quant is browser-incompatible), J1 parity gate PASS (worst diff 0.0079)
- [x] Days 8–9 — training: 6 runs, winner weighted-CE lr 1e-3, val macro-F1 0.3061 (majority floor 0.1813)
- [x] Days 10–11 — eval artifacts (validation), trained int8 shipped (60 KB, Δ macro-F1 −0.0016), Next.js app + fake-webcam e2e PASS
- [x] Days 14–15 — **final test-set evaluation (run exactly once, 2026-08-02)**: fp32 macro-F1 **0.2475** vs majority floor 0.1655; int8 0.2460 (Δ −0.0015); 3-class merged 0.3318; J3 browser benchmarks archived
- [x] 2026-08-03 — Track A + Track B repositories merged (unrelated histories); real trained model kept, Azeem's app canonical in `web/`
- [x] 2026-08-03 — CONTRACT.md v1.1 (§6 Amendment 1: 3 s prediction cadence), both partners signed; multi-face detection (up to 4, primary-face prediction, People count); landmarker pinned to CPU delegate per parity evidence
- [x] 2026-08-09 — Gap-closure pass: J1 rebuilt as a real Playwright Test (caught and fixed a real `numFaces:4` parity regression — see CONTRACT.md Amendment 2, v1.2); A5 baselines (`docs/results/baselines.csv`); J3 benchmark collector rewritten + re-run; J2 e2e script fixed and re-run; CI added (`.github/workflows/ci.yml`)
- [ ] Thesis writeup: methodology, error analysis (class 0 has only 4 test clips; adjacent-class confusion dominates)
- [ ] Two more J3 benchmark machines (runbook: `docs/benchmarks/README.md`)

Full day-by-day record: `docs/PROGRESS.md`.

## Headline results

| Metric (Test, 14,241 windows) | Value |
|---|---|
| Macro-F1 (fp32) | 0.2475 |
| Macro-F1 (int8, shipped) | 0.2460 |
| Majority-class floor | 0.1655 |
| 3-class merged macro-F1 | 0.3318 |
| Model size fp32 → int8 | 163 KB → 60 KB |
| In-browser inference (p50) | 0.4 ms |
| Feature parity, Python↔browser | max diff 0.0079 (tol 0.02) |
