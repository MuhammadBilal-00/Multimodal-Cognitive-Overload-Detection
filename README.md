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
- [ ] DAiSEE extracted to `data/DAiSEE/` with official Train/Validation/Test folders
- [ ] Audio-stream check on first clip (see CONTRACT.md §8)
- [ ] Day 2 — `features.py` + visual landmark verification
