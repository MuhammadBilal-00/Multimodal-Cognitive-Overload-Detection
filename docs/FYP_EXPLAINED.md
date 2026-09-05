# The Complete FYP Explainer

*A single document that explains this entire project from zero — every model,
every pipeline stage, every design decision, with real code from the repo —
plus the points that matter most for marking and every question a viva panel
is likely to ask. Written so that a beginner can follow it, and so that you
can teach it to someone else, which is the best test of whether you own it.*

**How to use it:** read §1–§9 once end-to-end to load the whole story into
your head; memorise the table in §10; then drill §12 (the Q&A) out loud until
the answers come without looking. Everything here cites the real file it
comes from — if a panel member opens the repo, the code will say what you
said.

---

## 1. The 60-second story

Online learning removed something classrooms always had: a teacher glancing
around the room to see who is engaged and who is lost. Research has tried to
restore that signal by pointing algorithms at learners' webcams — but almost
all of it ships the video (or detailed facial data) to a server, which for a
system whose whole job is *continuously watching a face* is an enormous,
usually hand-waved privacy cost.

This project asks: **can a useful engagement classifier run entirely inside
the learner's own browser — nothing leaving the machine, provably — fast
enough for live use, on ordinary hardware?** And it answers with a complete
working system, not a mock-up:

- video → **MediaPipe** finds 478 facial landmark points per frame (in the
  browser, via WebAssembly);
- 13 hand-designed geometric numbers per frame are computed from those
  points (eye openness, mouth openness, brow raise, head pose, gaze);
- 3 seconds of those numbers (30 frames at 10 Hz) go into a tiny neural
  network — a **Temporal Convolutional Network (TCN)**, 41,544 parameters,
  shrunk to a **60 KB int8 ONNX file** — which outputs an engagement level
  (Very Low / Low / High / Very High) and four independent state likelihoods
  (bored, engaged, confused, frustrated);
- a dashboard shows the trend over the last 60 seconds.

The three claims everyone else *asserts*, this project *measures*:
1. "The browser computes the same features as training" → an automated
   cross-language **parity gate** proves it numerically (and caught two real
   bugs when it drifted).
2. "No data leaves the device" → a **recorded network trace** proves it, and
   caught a real undisclosed telemetry call inside a Google library, which
   is now blocked by browser policy.
3. "The model is as good as this approach allows" → a **40-trial search
   over architectures, hyperparameters, features and ensembles** proves the
   shipped model sits at its representation's ceiling.

---

## 2. Foundations — the minimum you need to understand everything else

Skip this section if you know ML; teach from it if your listener doesn't.

**Supervised classification.** We have examples (3-second windows of facial
features) with human-provided answers (engagement level 0–3). The model
learns a function from example → answer on *training* data, and we check
whether it generalises on data it never saw.

**Train / Validation / Test splits.** Train = what the model learns from.
Validation = held-out data used to *choose between* models and settings.
Test = touched **once**, at the very end, to report the final number — if
you peek at it while making decisions, your final number is a lie. This
project used Test exactly once (2026-08-02) and never again for any decision.
Crucially, DAiSEE's official splits are **subject-independent**: no person
appears in two splits, so the model can't cheat by recognising faces.

**Class imbalance, and why accuracy lies.** In DAiSEE, ~95% of windows are
"engaged" or "very engaged". A useless model that always answers "engaged"
scores ~50–57% *accuracy*. So this project's primary metric is **macro-F1**:
compute the F1 score (harmonic mean of precision and recall) for each class
separately, then average them *equally* — a model that ignores the rare
classes gets punished. Reference floor: the always-majority model scores
macro-F1 **0.1655** on Test; anything meaningful must beat that.

**Logits, softmax, sigmoid.** A neural network's raw outputs are unbounded
numbers ("logits"). **Softmax** turns a set of logits into probabilities
that sum to 1 — right for "which ONE of 4 levels is it?". **Sigmoid**
squashes each logit independently into 0–1 — right for four *independent*
yes/no questions ("is boredom present? is confusion present?"), which is
what the states head answers. This distinction caused a real shipped bug
(§6.3), so know it cold.

**Overfitting & early stopping.** Train too long and the model memorises the
training set. We watch validation macro-F1 after every epoch and keep the
best checkpoint, stopping after 15 epochs without improvement.

**Quantization.** Neural network weights are normally 32-bit floats.
Quantization stores them as 8-bit integers — ~4× smaller, at a tiny accuracy
cost. Here: 163 KB → 60 KB, macro-F1 cost −0.0015. (Surprise finding: it's
about the *size*, not speed — see §9.4.)

**ONNX & WebAssembly (WASM).** ONNX is a portable file format for trained
models — export from PyTorch, run anywhere. WASM lets near-native code (the
ONNX runtime, MediaPipe) run inside a browser tab. Together they're what
makes "the model runs client-side" physically possible.
 
 
---

## 3. The dataset — DAiSEE

**DAiSEE** (Gupta et al., 2016): 10-second webcam clips of 112 university
students watching educational videos; each clip crowd-labelled 0–3 on four
affective states — Boredom, Engagement, Confusion, Frustration.

The numbers that matter (know these exactly — they were once wrong in the
abstract and got fixed, which a panel may probe):

| Fact | Value |
|---|---|
| Clips on disk (official Train/Val/Test folders) | 9,067 (5,481 / 1,720 / 1,866) |
| Clips with BOTH a label and a successful extraction (used) | **8,570** (5,357 / 1,429 / 1,784) |
| Windows produced (30-frame, stride 10) | 42,856 / 11,432 / 14,241 |
| Unique subjects in Train | 69 (≈78 clips each — this matters in §8.1!) |
| Rarest class on Test | class 0 "very low": **32 windows from 4 clips** |
| Face-detection rate during extraction | 99.96% |
| Audio | **None** — checked with ffprobe on day 1; "multimodal" in this project means three *visual* families (geometric, pose, gaze) |

The 4-clip rarest class is the single most important dataset fact: it makes
class-0 metrics statistically meaningless on their own, drives the choice of
macro-F1, and explains the below-chance class-0 AUC on the Test ROC (§12,
Q3).

---

## 4. The pipeline, stage by stage (with the real code)

### 4.1 Landmarks → 13 features (`ml/src/features.py`)

MediaPipe's FaceLandmarker returns 478 (x, y, z) points per frame. We reduce
them to 13 interpretable numbers. The most famous one, **Eye Aspect Ratio
(EAR)** — from Soukupová & Čech (2016) — measures eye openness from six
landmarks:

```python
# ml/src/features.py — EAR = (|p2−p6| + |p3−p5|) / (2·|p1−p4|)
vertical = _distance(p2, p6) + _distance(p3, p5)   # two eyelid gaps
horizontal = _distance(p1, p4)                     # eye corner-to-corner
return vertical / (2.0 * horizontal + EPS)
```

Open eye ≈ 0.3, blink → drops toward 0. Because it's a *ratio*, it doesn't
matter how close you sit to the camera.

The full 13 (order is FROZEN by the contract):

| # | Feature | What it captures |
|---|---|---|
| 0–2 | `ear_left`, `ear_right`, `ear_mean` | eye openness / blinking |
| 3 | `mar` | mouth openness (yawns, talking) |
| 4–5 | `brow_left`, `brow_right` | eyebrow raise ÷ interocular distance |
| 6 | `yaw` | head turned left/right (geometric proxy) |
| 7 | `pitch` | head up/down (proxy, centred on the training mean) |
| 8 | `roll` | head tilt (a true angle, from the eye-corner line) |
| 9–10 | `gaze_x`, `gaze_y` | iris offset within the eye |
| 11 | `face_area` | landmark bounding box ÷ frame area |
| 12 | `face_present` | 1.0 if a face was found, else the frame is 13 zeros |

Two frozen rules to be able to recite: **every raw distance is divided by
the interocular distance** (outer eye corner to outer eye corner) so the
features are camera-distance invariant — DAiSEE laptops and your webcam
become comparable; and **a frame with no face emits thirteen zeros — never
interpolate, never drop** — so both sides of the pipeline behave identically
when you look away.

Head pose is deliberately NOT solved with the standard `cv2.solvePnP`: its
browser equivalent (opencv.js) costs ~8 MB of WASM. Instead, three geometric
*proxies* (e.g. yaw = the asymmetry of nose-to-eye-corner distances) —
monotonic in the true angle, which is all a learned model needs, and stated
as proxies everywhere.

### 4.2 Windowing + standardisation (`ml/src/dataset.py`)

Clips are sampled at 10 FPS (every 3rd frame of 30 FPS video) → ~100 feature
rows per clip. A sliding window of **30 frames (3.0 s), stride 10** turns
each clip into ~8 training examples, each inheriting the clip's label.

```python
# ml/src/dataset.py — the split-integrity assert (not a comment, an assert)
for a in SPLITS:
    for b in SPLITS:
        if a < b:
            assert not set(usable[a]) & set(usable[b]), f"clip overlap {a}/{b}"
```

Then a scaler (mean/std per feature) is fitted **on Train only** and applied
everywhere — including in the browser, via the shipped `scaler.json`. The
`face_present` flag gets identity scaling (mean 0, std 1) so the binary flag
passes through untouched.

### 4.3 The model — a Temporal Convolutional Network (`ml/src/model.py`)

For a beginner: a *convolution over time* slides a small pattern-detector
along the 30-step sequence. **Dilated** convolutions skip steps (dilation 2
looks at every 2nd frame, dilation 8 every 8th), so four stacked layers with
dilations 1, 2, 4, 8 let the top layer "see" the whole 3-second window while
staying tiny. A **residual connection** (add the block's input to its
output) keeps training stable.

```python
# ml/src/model.py — one block, and the whole model
class TCNBlock(nn.Module):
    """Conv1d(k=3, dilated, same-padding) + BN + ReLU + Dropout + residual."""
    def forward(self, x):
        out = self.dropout(self.relu(self.bn(self.conv(x))))
        return out + self.residual(x)

class EngagementTCN(nn.Module):
    def forward(self, x):                      # x: (batch, 30, 13)
        h = x.transpose(1, 2)                  # → (batch, 13, 30) for Conv1d
        h = self.blocks(h)                     # 4 dilated blocks, 64 channels
        h = h.mean(dim=2)                      # average over time (exports cleanly)
        return self.head_engagement(h), self.head_states(h)   # (B,4), (B,4)
```

**41,544 parameters**, asserted `< 100_000` in the file itself — the entire
edge-deployment argument rests on the model staying this small. Two output
heads: 4 engagement logits (softmax → one level) and 4 state logits
(sigmoid → four independent likelihoods).

Why a TCN and not an LSTM? Bai, Kolter & Koltun (2018) showed TCNs match or
beat recurrent networks for sequence tasks with better parallelism and
stability — and the closest prior DAiSEE work (Abedi & Khan, 2021) found the
same on this dataset. We later verified it on *our* features too (§5).

### 4.4 Training (`ml/src/train.py`)

```python
# The loss: class-weighted CE for engagement + weighted BCE for states
loss = (engagement_loss(logits_eng, y_eng)          # CrossEntropy, inverse-
        + args.state_loss_weight                    #  frequency class weights
        * states_loss(logits_states, y_states))    # BCE with pos_weight
```

Key decisions, each defensible in one line:
- **Inverse-frequency class weights** — without them the 0.6%-of-training
  class 0 contributes almost no gradient and the model never learns it.
- **Early stopping on validation *macro-F1*** — not loss, not accuracy —
  because macro-F1 is the metric we actually care about.
- **pos_weight on the states head** — the original unweighted version
  *collapsed*: confusion/frustration (11.6%/7% prevalence) just predicted
  their base rates (AUC ~0.53 — chance). Retrained with per-channel
  neg/pos weighting; recall went 0% → 54%/48%.
- Six configurations were tried originally (focal loss, lower LR, sqrt
  weights, label smoothing) — plain weighted CE at lr 1e-3 won:
  **validation macro-F1 0.3061** vs the 0.1813 floor.

### 4.5 Export + quantization (`ml/src/export.py`)

```python
# The gate that stops a broken export shipping: PyTorch vs ONNX on
# 100 random inputs must agree to 1e-5 — or the build aborts.
if worst >= PARITY_TOLERANCE:
    raise SystemExit("PARITY CHECK FAILED — build aborted")

quantize_static(str(fp32_path), str(int8_path), WindowReader(),   # STATIC QDQ,
                quant_format=QuantFormat.QDQ, ...)                # calibrated on
                                                                  # real training windows
```

An empirical finding worth telling: the first attempt used *dynamic*
quantization — it worked in Python and **failed in the browser**, because
dynamic quantization emits a `ConvInteger` operator that onnxruntime-web's
WASM backend doesn't implement. Static QDQ emits `QLinearConv`, which it
does. Caught by a dedicated browser smoke test *before* shipping — the
project's recurring theme: "works in Python" and "works where it's deployed"
are different claims, so test the deployed one.

### 4.6 The browser side (`web/`)

The TypeScript app mirrors the pipeline exactly. The core loop
(`web/hooks/usePipeline.ts`): a requestAnimationFrame loop samples the video
at a true 10 Hz (an accumulator clock — see §6.4), runs the landmarker,
computes the same 13 features (`web/lib/features.ts`, a line-for-line port),
pushes them into a ring buffer:

```ts
// web/lib/ringBuffer.ts — the 3-second window, oldest frame falls out
push(f: Float32Array): void {
  this.buf.copyWithin(0, FRAME);                 // shift everything left
  this.buf.set(f, (WINDOW - 1) * FRAME);         // newest frame at the end
  this.count++;
}
```

…and every 30th sample (one prediction per non-overlapping 3 s window) runs
inference:

```ts
// web/lib/inference.ts — standardise OUTSIDE the model, then the two heads
const std = standardise(win, scaler.mean, scaler.std);
const tensor = new ort.Tensor('float32', std, [1, 30, 13]);
const out = await session.run({ features: tensor });
return {
  engagement: softmax(out.engagement.data),   // ONE level → softmax
  states: sigmoid(out.states.data),           // 4 INDEPENDENT states → sigmoid
  ...
};
```

Two design details that are viva gold:

**Two separate landmarkers.** The overlay shows up to 4 faces
(`numFaces: 4`) but the model is fed ONLY by a `numFaces: 1` landmarker —
because measurement showed `numFaces: 4` shifts landmark positions enough to
break parity on blink frames (gaze_y error 0.86 vs 0.016!). Training used
`num_faces=1`, so inference must too. Costs a second WASM detection pass per
frame; correctness bought with compute.

**The channel-order contract, made executable** (`web/lib/states.ts`):

```ts
// Index N here MUST equal index N of prediction.states. NOT alphabetical —
// fixed by ml/src/labels.py LABEL_COLS. (A previous version listed these
// alphabetically ... the "Confused" bar was really showing P(engagement).)
export const STATE_CHANNELS = [
  { key: 'boredom', ... }, { key: 'engagement', ... },
  { key: 'confusion', ... }, { key: 'frustration', ... },
] as const;
```

A vitest test literally **opens `ml/src/labels.py` and parses it with a
regex** to check the JS order matches the Python order — a documentation
convention turned into an executable guard, after the bug it describes
actually shipped once (§6.3).

---

## 5. Every model in the project

| Model | What it is (one line for a beginner) | Where | Validation macro-F1 | Role |
|---|---|---|---|---|
| **TCN** (shipped) | stack of dilated 1-D convolutions over the 3 s window | `model.py` | 0.3061 (0.3015 ± 0.005 over 3 seeds) | the deployed model |
| **LSTM** | recurrent net that carries a memory cell step-to-step | `model.py` (`--arch lstm`) | 0.2938 ± 0.014 | comparison |
| **GRU** | lighter cousin of the LSTM | `model.py` (`--arch gru`) | 0.2907 ± 0.006 | comparison |
| **Transformer** | self-attention: every timestep looks at every other | `model.py` (`--arch transformer`) | 0.3019 ± 0.007 — **statistically tied with the TCN** | comparison |
| **CORAL ordinal TCN** | reformulates "which of 4 *ordered* levels" as 3 rank questions | `model.py` (`--ordinal`) | 0.2571 (but accuracy +0.039, QWK +0.026) | trade-off, not a win |
| **Logistic regression** | linear classifier on 65 summary stats per window | `baselines.py` | 0.2420 | classical baseline |
| **Random forest** | many decision trees voting | `baselines.py` | 0.2669 | classical baseline |
| **Gradient boosting** | trees built sequentially, each fixing the last's errors | `baselines.py` | **0.2907 — and 0.2910 on Test, significantly ABOVE the TCN's 0.2475 (p<0.001)** | the strongest classical baseline — know this cold |
| **Stacked ensemble** | a meta-model combining TCN+RF+GBM probability outputs | `ensemble_stack.py`, `ensemble_fix.py` | 0.3114 clip-level after repair — only *ties* the TCN | diagnosed failure (§8.3) |

The classical baselines don't see the raw sequence — each 30×13 window is
collapsed to 65 numbers (mean/std/min/max/range of each feature) first.
That's the point of having them: they measure how much the *temporal
ordering* is actually worth.

**The uncomfortable headline you must own, not hide:** gradient boosting
beats the shipped TCN on Test. The four-part answer for why the TCN still
ships is in §12 Q2 — practice it until it's reflexive.

---

## 6. The verification story — and the two real bugs it caught

This is the project's actual thesis: *treat "the browser computes what
training computed" as a measured property, not an assumption.* The mechanism
is `CONTRACT.md` — a frozen written spec both partners sign — plus the **J1
parity gate**: 100 real DAiSEE frames pushed through BOTH implementations,
every feature compared per frame, tolerance 0.02.

### 6.1 Bug #1 — the numFaces regression (caught 2026-08-09)
A UI feature (multi-face overlay) changed the landmarker to `numFaces: 4`.
Rebuilding the parity gate to test the *production* landmarker (not a
hand-configured copy) instantly failed at gaze_y = **0.86** on blink frames
— 40× tolerance. The model had silently been fed features from a
configuration it was never trained on: textbook train/serve skew, invisible
to any offline evaluation. Fix: the two-landmarker split (§4.6). Recorded as
Amendment 2.

### 6.2 Bug #2 — the brow formula divergence (caught 2026-08-29)
Subtler and the better story: inside the *passing* parity report, the two
brow features' diffs (~0.011) sat **18–155× above every other feature's** —
all computed from identical landmarks on identical frames. Runtime noise
cannot be feature-selective; a formula can. The TypeScript port had computed
the brow's "eye centre" as the centroid of all six eye landmarks; the Python
reference uses the **midpoint of the two eye corners**. The eyelids drag the
centroid off the corner line → a systematic ~0.39σ bias on 2 of the model's
13 inputs at inference. Two documents had recorded the anomaly and
rationalised it as "expected noise". Fixed (Amendment 4), pinned by a
deliberately *asymmetric-eyelid* unit test (the symmetric fixture can't tell
the two formulas apart), and the gate's worst-case diff fell **0.0157 →
0.0035**. Transferable lesson to quote: *"a feature-selective parity
residual is a formula bug until proven otherwise."*

### 6.3 Bug #3 — the states channel swap (caught 2026-08-14)
The UI listed states alphabetically (Bored, Confused, Engaged, Frustrated);
the model emits Boredom, **Engagement, Confusion**, Frustration. Indices 1
and 2 were swapped, so the "Confused 99%" bar was actually showing
P(engagement). Fix: `states.ts` as the single mapping site + the
Python-source-parsing test (§4.6). Recorded as Amendment 3.

### 6.4 Smaller audit catches (know they exist)
The sampling loop originally drifted to ~8.6 Hz (display-refresh
quantisation) → fixed with an accumulator clock, and the live HUD now reads
a true 10 Hz. The ring buffer now resets across pauses so a "3-second
window" can never silently span a benchmark run or hidden-tab gap. `gaze_y`
divides by eye height, so a mid-blink frame can spike to ~26σ — documented
as a limitation (fixing it needs a retrain; a one-sided clamp would itself
create skew).

**What runs where** (be precise — the report is): unit/contract suites (38
Python + 41 TS tests + typecheck) and the export-parity gate run in CI on
every push; the J1 gate runs on any machine with the licence-restricted
DAiSEE fixture (it can't legally be committed, so hosted CI skips it with a
visible warning); browser smoke and e2e are release-point scripts.

---

## 7. The privacy story

Three layers, escalating from claim to proof:

1. **Architecture**: every asset self-hosted (model, WASM, landmarker file);
   no CDN, no third-party origin; camera frames never leave the `<video>`
   element and the WASM heap.
2. **Policy**: `next.config.mjs` ships a full CSP lockdown — `default-src
   'self'`, plus `Permissions-Policy: camera=(self), microphone=()`,
   COOP/COEP (which also unlock multi-threaded WASM — the wasm×20 in the
   HUD). Precision matters: the original single `connect-src` directive
   only governs fetch-type calls; the audit corrected the wording and the
   policy now closes image/script/form/frame routes too.
3. **Evidence**: `ml/scripts/privacy_trace.py` — a committed script that
   starts the production build, runs a fake camera for 75 s, and records
   every network request. Result: 39 requests, all same-origin, **zero
   external** — and the star exhibit: `@mediapipe/tasks-vision` (Google's
   own library) tries to POST usage telemetry to `odml.pa.googleapis.com`
   every ~60 s, undocumented, with no opt-out. The trace captures the CSP
   *blocking it*, twice, verbatim. The script **fails loudly** on any
   external request — the privacy claim regenerates on demand.

The one-liner: *"We caught a Google library phoning home from inside our
privacy-preserving app, blocked it at the browser-policy level, and kept the
receipt."*

---

## 8. The rigorous search — proving the ceiling

After the honest results were in (macro-F1 ~0.31 validation), the obvious
attack is "you just didn't tune it well." So the project eliminated every
version of that critique, and the *failures* here are the findings:

### 8.1 A bug caught in our own statistics (tell this proudly)
The first cross-validation design grouped folds by **clip**. But Train has
5,357 clips from only **69 people** (~78 clips each) — so the same person's
clips landed on both sides of a fold, and the model could recognise *faces*
instead of *engagement*. That leaky setup produced a beautiful, clean
finding (a pose-free feature subset "beating" the full set with
non-overlapping ranges)… which **evaporated completely** when folds were
regrouped by subject and everything re-run:

```python
# ml/src/cv_splits.py — the fix, and the test that keeps it fixed
def subjects_of(clip_ids):        # first 6 chars = DAiSEE subject ID
    return np.array([str(c)[:6] for c in clip_ids])

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
yield from skf.split(np.zeros(len(y)), y, groups=subjects)   # by SUBJECT
```

After the fix: **no feature subset differs significantly** (all p > 0.11).
An evaluation-design error that manufactured a publishable-looking result,
caught before it reached any conclusion. The marking scheme literally awards
"identified mistakes made and lessons learnt" — this is that, with receipts.

### 8.2 Hyperparameter search (Optuna)
40 trials of Bayesian search (TPE sampler, median pruning — half the trials
were cut early) over learning rate, width, dropout, weighting, batch size.
The winner, retrained at full budget over 5 seeds: mean 0.3085 vs the
shipped 0.3015 — and a clip-level bootstrap says the difference is noise
(**p = 0.92**). The shipped configuration — originally chosen from six
manual runs — is *validated*, not lucky.

### 8.3 The ensemble that failed, diagnosed to the bone
Stack TCN + random forest + gradient boosting probabilities into a logistic
regression meta-model (proper out-of-fold construction, zero leakage).
First result: **worse than every single input** — the ensemble never
predicted the two rare classes at all. Diagnosis: the meta-learner weighted
the random forest 3.6× the TCN, and RF (despite balanced weighting) makes
almost no rare-class predictions; an unweighted meta-learner trained on 95%
majority data learns to trust confident majority votes and discard the
TCN's noisy-but-real rare-class signal. A fully "balanced" meta-learner
over-corrects and collapses (imbalance corrections compose badly — they
double-count). The repaired version (drop RF, moderate weighting) recovers
to 0.3114 — which only *ties* the TCN. Ensembling: investigated, diagnosed,
closed.

### 8.4 Everything else tried, honestly reported
Augmentation (noise + temporal masking, 3 seeds): 0.2969 ± 0.008 — no help.
CORAL ordinal head: trades macro-F1 for accuracy/QWK — a different point on
the metric curve, not a better one.

**The conclusion all of this buys:** the limitation is the **13-feature
representation itself** — not undertuning, not the wrong architecture, not
an unexploited ensemble. Which converts "future work: a learned
representation" from a wish into the *only remaining lever*, justified by
elimination.

---

## 9. Honest evaluation — measuring on the right basis

### 9.1 Clip-level scoring
DAiSEE's labels are per-clip and published benchmarks report per-clip — but
this project had only ever scored per-window (~8 overlapping windows per
clip). Averaging each clip's window probabilities makes the numbers
commensurable with the literature for the first time.

### 9.2 Threshold calibration (the best cheap win in the project)
Four per-class additive offsets, tuned on Validation only (coordinate
ascent on clip-level macro-F1), then applied **frozen** to the Test
predictions:

```python
# ml/src/clip_eval.py — decision-layer calibration, never tuned on Test
offsets = tune_offsets(clip_probs, clip_labels)     # on VALIDATION
tuned_pred = (np.log(clip_probs + 1e-12) + offsets).argmax(1)
```

Result: **Test macro-F1 0.2475 → 0.2829, accuracy 36.9% → 44.7%** — a +14%
relative improvement from four constants, out-of-sample, with the deployed
model untouched. (Shipping those four constants to the browser is
future-work item #1.)

### 9.3 The binary-accuracy trap (a worked example to teach with)
Collapse to "disengaged vs engaged" and the model scores ~78% accuracy —
sounds great, except **always answering "engaged" scores 88–95%** at this
prevalence. The honest numbers are AUC 0.64/0.68 and balanced accuracy
~0.60–0.62: modest, real, above chance. This is the whole
accuracy-under-imbalance lesson in one table, and it's why the project never
headlines accuracy.

### 9.4 The quantization latency surprise
int8 is ~9% *slower* than fp32 in native onnxruntime at this model size —
the dequantise/requantise overhead outweighs integer gains on 41k
parameters. Quantization's value here is the 60 KB payload and WASM operator
compatibility, **not speed**. Volunteering this before being asked reads as
mastery.

---

## 10. The numbers to memorise

| Number | What it is |
|---|---|
| **0.3061 / 0.3043** | validation macro-F1 (frozen ckpt / shipped ckpt — states retrain; disclosed in §4.3.2 of thesis) |
| **0.2475 → 0.2829** | Test macro-F1, window-level → clip-level calibrated |
| **36.9% → 44.7%** | Test accuracy, same reframing |
| **0.1655 / 0.1813** | majority-class macro-F1 floor (Test / Validation) |
| **0.2910, p<0.001** | gradient boosting on Test — significantly beats the TCN |
| **p=0.194 / p=0.92** | TCN vs GBM on Validation (tie) / tuned vs shipped TCN (tie) |
| **63.9%** | published SOTA accuracy (Abedi & Khan 2021, ResNet+TCN, GPU, not deployable) |
| **41,544 / 60 KB / 163 KB** | TCN parameters / int8 size / fp32 size |
| **0.0035 (tol 0.02)** | parity gate worst-case diff after the brow fix (was 0.0157) |
| **0.86 vs 0.016** | gaze_y parity at numFaces 4 vs 1 — why two landmarkers |
| **36 fps / 10 Hz / 0.78 ms / wasm×20** | your OWN live rehearsal readings, 2026-08-29 |
| **8,570 clips (9,067 on disk)** | training corpus / total extracted |
| **4 clips / 32 windows** | class 0 on Test — the statistical black hole |
| **75 s, 0 external requests** | the privacy trace |

---

## 11. What the markers actually grade (focus here)

The marking scheme has four equally weighted sections. Map your talking
points onto them deliberately:

1. **Understanding of the problem domain** → the privacy framing as an
   engineering constraint (not a consent form); the literature positioning
   (§2.7 of the thesis — the unoccupied niche); the class-imbalance analysis
   driving every metric choice.
2. **Development of product and ideas** → alternatives genuinely considered
   with reasons (PnP vs geometric proxies; SMOTE vs loss weighting;
   dynamic vs static quantization — each rejected on *evidence*); the
   contract-driven two-track methodology itself.
3. **Product build and evaluation** → the working system (demo!), the
   parity/export/e2e gates, the breadth of §5's model comparison, the
   statistical rigour of §8, the honest measurement of §9.
4. **Conclusions and critical review** → the crown jewels: the three caught
   bugs, the subject-leakage correction, the ceiling conclusion, the
   evidence-ordered future work. The ≥80% band asks for "a high standard of
   critical analysis" — this project's critical analysis is its strongest
   asset. **Lead with it, never apologise with it.**

Presentation criteria are pass/fail-ish: word count in band (14,997 ✓),
Harvard referencing (✓, every source verified real), figures labelled AND
referenced (✓ — every caption names its data file), demonstration held
(your job on the 3rd), Turnitin (yours).

---

## 12. The full viva Q&A

*(Say each answer out loud. The first six are the ones most likely to be
asked; the "beginner" block at the end is for concept checks a panel uses to
probe whether the work is really yours.)*

**Q1. Your accuracy is only ~44%. Why is this worth a Masters?**
Published SOTA on this exact task is 63.9% — from a GPU ResNet on raw video
that can never run privately in a browser. Random is 25%; always-majority
gets ~50% accuracy while being useless (macro-F1 0.17). The labels are noisy
crowd annotations of subtle distinctions — that caps what anyone can
measure. Our contribution is the privacy/deployability trade made *honest
and exact*: we know precisely what 60 KB of private, in-browser inference
costs in accuracy, and the app is designed around what that supports —
trends, never point judgements.

**Q2. Gradient boosting beats your model on Test. Why ship the TCN?**
(1) On Validation they're statistically level (p=0.194). (2) Gradient
boosting cannot ship — there is no browser-WASM path for a quantized tree
ensemble, and edge deployment is the premise, not a preference. (3) The TCN
beats LSTM/GRU at matched budgets and a 40-trial search couldn't improve it
(p=0.92) — its configuration is validated. (4) Therefore the ceiling is the
feature representation, which is exactly future-work item 4. And note: we
published this result against ourselves — `significance.json` is our file.

**Q3. Figure 4.3 shows class 0 with AUC 0.433 — worse than random?**
Thirty-two windows from four clips can't estimate an AUC — below 0.5 on
that sample means "unmeasurable", not "anti-predictive". Class 2's ~0.5 is
different: one-vs-rest scoring is structurally hard for the *middle band* of
an ordinal scale. Both are why macro-F1 over hard assignments is primary and
why the 3-class merged metric exists.

**Q4. Which checkpoint produced which numbers?**
Test was consumed exactly once (2026-08-02, frozen checkpoint, macro-F1
0.2475). The browser ships a later checkpoint whose *states head* was
retrained (the engagement head is within noise: 0.3043 vs 0.3061 val) —
deliberately never re-evaluated on Test to protect the once-only claim.
Disclosed in thesis §4.3.2.

**Q5. Didn't your celebrated parity gate miss a bug for weeks?**
Yes — and we tell that story on purpose. The brow formula divergence hid
inside a passing gate because the 0.02 tolerance (needed for real runtime
noise) was wide enough to mask a small systematic error. Our audit caught it
via the principle that runtime noise can't be feature-selective; the fix
dropped the worst-case diff 4.5×. Verification catching our own mistakes —
three times now — *is* the thesis.

**Q6. What did AI tools do in this project?**
Per the Declaration of AI Use: assisted code drafting/refactoring, ran and
logged experiments, and drafted report prose under my review. Every result
comes from committed code executed against the real dataset — nothing was
generated without execution. The design decisions, the contract with my
partner, and everything I'm saying today without notes are mine.

**Q7. Why 13 hand-crafted features instead of deep learning on pixels?**
Browser compute budget; interpretability; and — decisive — a documented
arithmetic formula can be implemented twice and *numerically verified*
across languages, which a CNN embedding cannot. The cost is the gap to SOTA,
quantified; the search proved the ceiling is this representation, so the
trade is now measured, not assumed.

**Q8. Why is "multimodal" in the title when there's no audio?**
Checked day one — DAiSEE has no audio stream (ffprobe, 9 clips, all
splits). From day one "multimodal" has meant three *visual* families:
geometric (eyes/mouth/brows), pose, gaze — stated in CONTRACT §8 and used
consistently. The ablation later showed their contributions are not simply
additive for the TCN — honest nuance in §4.8.

**Q9. What stops the model just recognising people instead of engagement?**
DAiSEE's official splits are subject-independent (asserted in code, not
assumed), and our internal cross-validation groups by subject too — after we
caught our own first CV design leaking subjects across folds and watched a
"finding" evaporate when we fixed it. That episode is in the thesis.

**Q10. Two people in frame — whose engagement is it?**
Honest limitation (§5.2.2c): the overlay's highlighted face and the
model-feeding single-face detector can disagree; they're separate *because*
parity demands numFaces:1. Single-learner use is the design target;
multi-learner needs reconciliation — future work.

**Q11. What happens when I leave the frame?**
Each frame emits thirteen zeros (frozen rule — never interpolate). A fully
empty window standardises to extreme values the model never saw (99.96%
training detection rate), so its output there is arbitrary — the UI detects
this (`ood.noFace`) and suppresses the readings entirely ("— no face"),
resuming within one window of return. Found and fixed in our own live
rehearsal.

**Q12. How do you know the browser computes the same features as training?**
We don't *know* it — we *measure* it: 100 real frames through both
implementations, 13 features each, tolerance 0.02, currently passing at
0.0035 worst-case. Plus an export gate (PyTorch↔ONNX at 1e-5, in CI) and a
source-parsing test pinning the states channel order to the Python source.

**Q13. What would you do with three more months?**
In evidence order: ship the four-constant calibration to the browser (+14%
relative Test macro-F1, already validated); the remaining benchmark
machines; a learned spatial representation — now justified by elimination;
an aperture-gated gaze_y redefinition (the 26σ blink finding); a larger
low-engagement evaluation set.

**Q14. Is this system ready to judge real students?**
No, and it doesn't claim to be — no system at 44% (or even SOTA's 63.9%)
should make consequential decisions about individuals. It's an aggregate,
trend-level signal with the privacy problem *solved by construction*, and
the interface is built to communicate exactly that level of confidence.

**Beginner concept checks (panels use these to test ownership):**

- *What is a TCN?* — Stacked 1-D convolutions over time with increasing
  dilation, so a small, parallel, stable network sees the whole window;
  ours: 4 blocks, dilations 1/2/4/8, 64 channels, residuals, mean-pool, two
  linear heads.
- *Softmax vs sigmoid — why both?* — Softmax for "which ONE level"
  (probabilities sum to 1); sigmoid for four *independent* "is this state
  present ≥ level 2?" questions (they don't sum to 1 — the UI even says so).
- *What is macro-F1 and why not accuracy?* — Per-class F1 averaged equally;
  at 95% majority prevalence, accuracy rewards ignoring rare classes and
  macro-F1 punishes it.
- *What is QWK?* — Quadratic weighted kappa: an agreement measure that
  penalises far-off errors more than adjacent ones — fits an *ordinal*
  label; it even reorders our model ranking, proving metric choice matters.
- *What is quantization, and what did it cost here?* — Float32 → int8
  weights: 163→60 KB, −0.0015 macro-F1, and (surprise) ~9% *slower* in
  native ORT — it's about size and WASM compatibility, not speed.
- *What is standardisation and where does it happen?* — (x−mean)/std with
  Train-fitted stats; deliberately *outside* the model graph, in the caller,
  identically in Python and JS via the shipped `scaler.json`.
- *Why 3-second windows at 10 Hz?* — Frozen contract: 30 frames covers a
  meaningful behavioural unit, keeps the model input small, and 10 Hz is
  every 3rd frame of DAiSEE's 30 fps; the live app shows one prediction per
  non-overlapping window for readability.
- *What is train/serve skew?* — Any difference between what the model saw
  in training and what it's fed in deployment; this project's central risk,
  central methodology, and the thing all three caught bugs have in common.

---

## 13. Glossary (fast lookups)

**CONTRACT.md** — the frozen interface spec both tracks build against; four
amendments, each documenting a decision or a caught bug. · **J1/J2/J3** —
the joint gates: feature parity / end-to-end app test / multi-machine
benchmarks. · **DAiSEE** — the dataset (§3). · **EAR/MAR** — eye/mouth
aspect ratios. · **Landmark** — one of 478 (x,y,z) facial points from
MediaPipe. · **Logit** — a model's raw pre-probability output. · **Macro-F1
/ QWK** — the two headline metrics (§12 beginner block). · **ONNX / QDQ /
WASM** — model format / static quantization style / browser bytecode
(§2, §4.5). · **OOD** — out-of-distribution; the dashboard's honesty layer.
· **Parity gate** — the Python↔TypeScript numerical equivalence test. ·
**Stride** — how far the sliding window moves between examples (10 in
training; 30 live = non-overlapping). · **Subject-independent** — no person
in two splits; the anti-face-recognition guarantee. · **TCN** — the shipped
model (§4.3).

---

*Companions: `docs/viva-pack.md` (timing, contingencies, do-not-say list),
`docs/dry-run-checklist.md` (the demo script + your recorded rehearsal),
and the thesis
itself — whose §5 says everything here in examiner-facing prose.*
