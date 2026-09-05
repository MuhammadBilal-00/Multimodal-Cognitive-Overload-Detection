# Student handoff — pipeline walkthrough

The handoff objective is: *"The student can walk the full pipeline unaided
and answer the questions in her prep plan."*

**This document is preparation material, not the handoff itself.** The
actual handoff is a live session with the student — walking
through the real code together, then confirming she can explain it back
unaided. Nothing here can substitute for that conversation; this just
means you don't have to build the walkthrough and the Q&A bank from
scratch in the room.

Because A5's baselines were implemented directly in the project, she should
understand `baselines.py` at the
same depth as if she'd written it — not just receive the CSV it produces.
**Checkpoint 4** is specifically: *she can run `python ml/src/baselines.py`
and explain every line of the CSV it produces, unaided, on request.* The
"Deep dive: baselines.py" section below is written to make that concrete.

---

## 1. The pipeline, stage by stage

Run these commands together, in order, reading the source alongside each
one — not slides made about the code, the actual code (per Phase 4's own
instruction).

### 1.1 Extraction — `ml/src/extract.py`

```powershell
python ml\src\extract.py --limit 5 --resume
```

**What it does:** for each DAiSEE clip, opens it with OpenCV, samples
every 3rd frame (30 fps source → 10 fps contract rate, `TARGET_FPS = 10.0`),
runs MediaPipe `FaceLandmarker` in `VIDEO` mode, converts the 478 landmarks
from normalized to pixel coordinates (`_landmarks_to_pixels`), and calls
`compute_features()` — writing one CSV row per sampled frame to
`artifacts/features/{split}/{clip_id}.csv`.

**Why it matters:** this is the ONLY place raw video ever gets touched.
Everything downstream (windowing, training, the web app) consumes the
13-float feature vectors this step produces — never the video itself.

**Key design point to walk through:** `multiprocessing.Pool` with a fresh
`FaceLandmarker` created *inside* each worker (`_init_worker`), not
shared. Ask her why a shared landmarker across clips would be wrong —
answer: `VIDEO` mode landmarkers expect monotonically increasing
timestamps and carry internal tracking state; reusing one across
unrelated clips would leak tracking state from clip N into clip N+1's
first few frames.

### 1.2 Features — `ml/src/features.py`

```powershell
python -m pytest ml\tests\test_features.py -v
```

**What it does:** the reference implementation of CONTRACT.md §2–4. Pure
functions, no I/O — `compute_features(landmarks, frame_shape, pitch_centre)`
takes one frame's 478 pixel-coordinate landmarks and returns the 13-float
feature vector (EAR × 2, MAR, brow × 2, yaw/pitch/roll, gaze × 2,
face_area, face_present).

**Why it matters:** this file IS the contract. `web/lib/features.ts` is a
line-for-line port of it. If she ever needs to explain why a feature is
defined the way it is, this file's own docstring (the five numbered
CONVENTIONS at the top) has the answer — have her read that docstring
aloud and explain each of the 5 points in her own words.

**Key design point:** every distance is divided by the interocular
distance (`INTEROCULAR = [33, 263]`) before use. Ask her why — answer:
scale invariance. A DAiSEE subject sitting far from a lab camera and a
live webcam user sitting close to their laptop produce very different
raw pixel distances for the same facial expression; dividing by
interocular distance (which scales with camera distance too) cancels
that out.

### 1.3 Windowing — `ml/src/dataset.py`

**What it does:** reads per-clip feature CSVs, joins them against DAiSEE's
label CSVs (`labels.py`), slides a 30-frame window (3.0 s) with stride 10
over each clip's ~100 rows (→ ~8 windows/clip), and writes
`artifacts/dataset/{Train,Validation,Test}.npz` — each containing
`x (N, 30, 13)`, `y_engagement (N,)`, `y_states (N, 4)`, `clip_ids (N,)`.
Fits a `StandardScaler` on **Train only**, applies it to all three splits,
saves `scaler.json`.

**Why it matters:** this is where "one video clip" becomes "many training
examples," and where the critical train/test hygiene lives — ask her to
find the assertion that guarantees zero clip-ID overlap between splits,
and explain in her own words why that assertion exists (a model that saw
part of a clip in training and another window from the *same* clip in
test would get an inflated, meaningless score — the model would be
recognizing the *person*, not the *engagement level*).

### 1.4 Baselines — `ml/src/baselines.py`

See the dedicated deep-dive below — this is the Checkpoint 4 focus.

### 1.5 Model + training — `ml/src/model.py`, `ml/src/train.py`

```powershell
python ml\src\model.py
```

**What it does:** `EngagementTCN` — 4 dilated causal Conv1d blocks
(dilations 1/2/4/8, each with BatchNorm+ReLU+Dropout+residual), global
average pool over time, two linear heads (engagement logits, state
logits). Running the file directly prints the parameter count (41.5k,
budget was <100k) and confirms the forward-pass shapes.

`train.py` trains it: Adam lr 1e-3, cosine schedule, batch 128, up to 100
epochs, early stopping on **validation macro-F1** (not loss, not
accuracy) with patience 15. Loss = `CE(engagement, inverse-freq class
weights) + 0.5 * BCE(states)`.

**Why it matters — this is one of the embedded prep-plan questions
(the embedded acceptance criterion):** DAiSEE's engagement levels are wildly imbalanced
— level 0 is ~0.7% of windows, level 1 ~4-5%. A model that always
predicts "high engagement" (the majority class) scores ~50-57% *accuracy*
while being completely useless — it never once correctly identifies a
disengaged student, which is the entire point of the system. **Macro-F1
averages per-class F1 equally**, so a model that ignores the minority
classes scores near-zero on it regardless of how high its accuracy looks.
That's why every acceptance criterion in this project (A7, A8, the
baselines) is measured in macro-F1, and why `train.py` uses
`inverse_frequency_weights()` in the loss — without it, gradient descent
has no incentive to ever predict the rare classes at all.

### 1.6 Export — `ml/src/export.py`

```powershell
python ml\src\export.py --checkpoint artifacts\runs\<timestamp>\best.pt
```
(without `--ship`, so it doesn't overwrite the shipped web model)

**What it does:** exports to ONNX (opset 17, dynamic batch axis),
`onnx.checker.check_model`s it, then — this is the load-bearing step —
runs 100 random inputs through **both** the PyTorch model and the
exported ONNX graph via onnxruntime and asserts max-abs-diff < 1e-5,
**aborting the build if it fails**. Only then does it quantize to int8
(static QDQ, not dynamic — ask her why: dynamic quantization emits
`ConvInteger` nodes, which onnxruntime-web's WASM backend doesn't
implement; this was found empirically during the A6.5 browser smoke, it's
not a guess).

**Why it matters:** this is a *second*, independent parity gate — J1
checks Python features vs. browser features; this checks PyTorch
predictions vs. ONNX predictions. Two different train/serve boundaries,
two different gates, both non-negotiable.

---

## 2. Deep dive: `baselines.py` (Checkpoint 4 focus)

```powershell
python ml\src\baselines.py
```

Walk through it top to bottom and have her explain each piece back:

1. **`aggregate_features(x)`** — takes a `(N, 30, 13)` array of windows and
   returns `(N, 65)`: for each of the 13 features, the mean, std, min,
   max, and range *across the 30 timesteps* (5 × 13 = 65). **Ask her why
   this exists at all**, given the TCN just eats the raw `(30, 13)`
   window directly: logistic regression and random forest have no notion
   of sequence order — they need one fixed-size feature vector per
   example, not a sequence. Collapsing the time axis into summary
   statistics is how you turn a window into something those model
   families can consume.

2. **The majority-class row** — `np.bincount(y_train, minlength=4).argmax()`
   gives the most common training-set class; every split's `majority` row
   is that constant prediction, scored the same way as every real model.
   **Ask her why this row is in the CSV at all**: it's the floor. A macro-F1
   of, say, 0.20 means nothing on its own — the question is always "0.20
   compared to what a model that learned nothing would score." The TCN's
   entire acceptance criterion is literally "beats
   this row."

3. **The `_3class_merged` rows** — same predictions, relabeled by
   collapsing engagement levels 0 and 1 into one "low" class before
   scoring. **Ask her why**: engagement level 0 has only 4 clips in the
   entire *test* split — its per-class metrics are
   statistically close to meaningless on their own. Merging 0+1 gives a
   secondary, still-honest metric that isn't dominated by that sparsity.
   This is the exact same collapse `eval.py` applies to the TCN's own
   results, for direct comparability.

4. **The output** — `docs/results/baselines.csv`, columns
   `split, model, macro_f1, accuracy, qwk` (the last is quadratic
   weighted kappa, an ordinal-aware companion metric added in Experiment
   7 — she should be able to say why an ordinal label makes plain
   macro-F1 incomplete). Have her open it and, for each row, say out loud
   which of the scenarios above it represents.

**Real numbers to check her against** (already produced, both splits):
majority-class macro-F1 is 0.1813 (Validation) / 0.1655 (Test) — and
these were cross-checked to match `metrics_{validation,test}.csv`'s own
majority-class row exactly, which is itself worth asking her to verify by
opening both files side by side. The TCN's validation macro-F1 (0.3061)
beats logreg (0.242) and random forest (0.2669) — **but she must know the
full, corrected story, because the panel will**: the strongest classical
baseline added later, gradient boosting, reaches 0.2907 on Validation
(the TCN's lead over it is NOT statistically significant — p=0.194,
`docs/results/significance.json`) and 0.2910 on Test, where it
*significantly beats* the TCN's 0.2475 (p<0.001). The honest claim is
therefore NOT "the TCN uses temporal structure the baselines can't and
therefore wins" — it is: the TCN is statistically level with the best
classical method on validation, loses to it on test, and is shipped
because it is the only one of the candidates that meets the deployment
constraint (a 60 KB int8 ONNX graph running in browser WASM — a tree
ensemble has no comparable path), with a 40-trial search later confirming
its configuration sits at the representation's ceiling
(`docs/results/rigorous_model_search.md`). If she can explain *that*
chain, she is ready for the hardest baseline question the panel can ask.

---

## 3. Prep-plan question bank

Straight from the project's acceptance criteria and design
rationale — these are the kinds of questions a panel is likely to ask,
because they're the kinds of questions the plan itself flags as
non-obvious.

**Q: Why macro-F1 instead of accuracy, throughout this entire project?**
A: DAiSEE's engagement classes are extremely imbalanced (class 0 ≈0.7% of
windows). A model that always predicts the majority class scores
50-57% *accuracy* while being useless for the actual task — it never
detects disengagement. Macro-F1 weighs every class equally regardless of
its size, so a model has to actually handle the rare classes to score
well on it.

**Q: Why did class weighting matter in training?**
A: Same imbalance. Without `inverse_frequency_weights()` in the loss,
gradient descent has no signal pushing it to ever predict the rare
classes — cross-entropy loss is dominated by the majority class by sheer
volume of examples.

**Q: What does the J1 parity gate actually prove, and why does it matter
this much?**
A: The model is trained on features computed by Python
(`ml/src/features.py`, via `extract.py`) but served by features computed
in the browser (`web/lib/features.ts`) — two independent implementations
of the same math, in two different languages, that must agree almost
exactly, or the model sees systematically different inputs live than it
saw during training ("train/serve skew"). J1 is the automated proof that
they agree, within a tolerance (0.02, CONTRACT.md Amendment 2) empirically
set from the actual observed Python-vs-browser MediaPipe landmark noise.
Rebuilding J1 properly in this session (2026-08-09) caught a real
instance of this exact failure mode: the app's multi-face landmarker
config (`numFaces:4`) was silently producing different — and on blink
frames, badly wrong — features than the single-face config the model was
trained on. That's not hypothetical; it's the reason the app now runs two
separate landmarker instances (`lib/faceLandmarker.ts`).

**Q: Why divide every distance by interocular distance?**
A: Scale invariance. DAiSEE subjects are recorded at a range of
distances from the camera; live webcam users sit at another range
entirely. Raw pixel distances for "the same" facial expression differ
enormously between the two. Dividing by interocular distance (itself
proportional to camera distance) cancels that scale factor out.

**Q: Why "geometric pose proxies" for yaw/pitch/roll instead of real
Euler angles via `cv2.solvePnP`?**
A: `solvePnP` needs a browser equivalent, and `opencv.js` costs ~8 MB of
WASM just for that one function — not worth it for an edge-deployment
story. The geometric proxies (CONTRACT.md §3) are monotonic in the true
angle and scale-invariant, which is all the model actually needs; they're
described as proxies, not calibrated angles, everywhere including the
thesis.

**Q: Why quantize to int8, and what does it cost?**
A: The edge-deployment story depends on the model being genuinely tiny
in-browser. Quantization took the model from 163 KB (fp32) to 60 KB
(int8) — 2.7× — for a measured macro-F1 cost of −0.0015 on Test
(`docs/results/quantization_test.csv`; the Validation-split version is
`quantization.csv`). One honest wrinkle to volunteer before being asked:
int8 is actually ~9% *slower* than fp32 in native onnxruntime at this
tiny model size (`docs/benchmarks/fp32_vs_int8_latency.json`) — the
QDQ dequantise overhead outweighs integer gains on a 41k-parameter net.
The quantization's real value is the payload size and the WASM operator
compatibility (dynamic quantization's ConvInteger doesn't exist in
onnxruntime-web), not speed.

**Q: Why is this project "multimodal" if DAiSEE has no audio?**
A: Checked explicitly on Day 1 (`ffprobe` on 9 clips across all splits) —
DAiSEE is video-only. "Multimodal" here means fusion of three distinct
*visual* modality families: geometric (EAR/MAR/brow), pose
(yaw/pitch/roll), and gaze (iris offset) — not audio+video. This framing
is used consistently everywhere, including the thesis, rather than
overclaiming.

---

## 4. Confirming the checkpoint

The handoff criterion is: *"she can run `python ml/src/baselines.py`
and explain every line of the CSV it produces, unaided, on request."*
Concretely — have her, without prompting:

1. Run the command and let it finish.
2. Open `docs/results/baselines.csv` and, for any row you point to, state
   which split, which model, and whether it's a plain or 3-class-merged
   score.
3. Explain what the `majority` row represents and why it's there.
4. Explain, in her own words, why the baselines use 65-dim aggregate
   features while the TCN uses the raw `(30, 13)` window.
5. Answer at least two of the §3 question-bank items unprompted.

This document doesn't get to mark that checkpoint done — only that
conversation does.
