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
- [ ] A5 baselines (reserved for the FYP student — inputs ready in `artifacts/dataset/`)
- [ ] Thesis writeup: methodology, error analysis (class 0 has only 4 test clips; adjacent-class confusion dominates), CONTRACT.md sign-offs

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
- [ ] Day 2 — `features.py` + visual landmark verification
