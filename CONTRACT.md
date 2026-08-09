# INTERFACE CONTRACT — Track A (ML) ↔ Track B (Web)

**Version 1.2 — Day 1; §6 Amendment 1 on 2026-08-03; Amendment 2 on 2026-08-09**
**Status of each section: FROZEN unless marked OPEN.**

This document is the single source of truth for everything that crosses the
boundary between the Python ML pipeline (Bilal, Track A) and the Next.js
in-browser inference app (Azeem, Track B). `ml/src/features.py` is the
reference implementation of Sections 2–4; Azeem ports it to TypeScript
literally. Changes to any FROZEN section require sign-off from both partners
and a version bump.

---

## 1. Deliverables crossing the boundary

| File | Producer | Consumer |
|---|---|---|
| `web/public/model/model_int8.onnx` | Track A | Track B |
| `web/public/model/scaler.json` | Track A | Track B |
| `ml/tests/fixtures/parity_clip.mp4` | Track A | Track B |
| `ml/tests/fixtures/parity_expected.json` | Track A | Track B |

Until the real model lands (Day 11), Track B builds against a dummy ONNX
model with the same I/O signature (Section 5) and a `scaler.json` with
`pitch_centre: 0.5` as placeholder.

---

## 2. Feature vector — 13 floats per frame, order FROZEN

| # | Name | Definition |
|---|---|---|
| 0 | `ear_left` | Eye Aspect Ratio, left eye |
| 1 | `ear_right` | Eye Aspect Ratio, right eye |
| 2 | `ear_mean` | mean of features 0 and 1 |
| 3 | `mar` | Mouth Aspect Ratio (vertical ÷ horizontal lip distance) |
| 4 | `brow_left` | left eyebrow-to-eye-centre distance ÷ interocular distance |
| 5 | `brow_right` | right eyebrow-to-eye-centre distance ÷ interocular distance |
| 6 | `yaw` | geometric yaw proxy, ~−1..1 |
| 7 | `pitch` | geometric pitch proxy, ~−1..1 |
| 8 | `roll` | true roll angle, radians |
| 9 | `gaze_x` | iris centre x offset ÷ eye width |
| 10 | `gaze_y` | iris centre y offset ÷ eye height |
| 11 | `face_area` | face bounding-box area ÷ frame area |
| 12 | `face_present` | 1.0 if a face was detected, else 0.0 |

**EAR formula** (landmarks p1..p6 per Section 4):

```
EAR = (|p2 − p6| + |p3 − p5|) / (2 · |p1 − p4|)
```

### Normalisation rule (FROZEN)

Every raw distance is divided by the **interocular distance** (outer eye
corner to outer eye corner, landmarks 33 ↔ 263) before any further use.
This is what makes DAiSEE faces (far from camera) and live webcam faces
(close to camera) comparable. Ratios that already have their own
denominator (EAR, MAR, gaze offsets, yaw, pitch) are inherently
scale-invariant and are not divided again.

### Missing-face rule (FROZEN)

If no face is detected in a frame: emit **thirteen zeros** with
`face_present = 0.0`. **Never interpolate. Never drop the frame.**

---

## 3. Head pose — geometric proxies (FROZEN)

`cv2.solvePnP` is **not used** (no clean browser equivalent; opencv.js
costs ~8 MB of WASM). All three angles are pure landmark geometry:

```python
roll = np.arctan2(right_eye_outer[1] - left_eye_outer[1],
                  right_eye_outer[0] - left_eye_outer[0])

dL = np.linalg.norm(nose_tip[:2] - left_eye_outer[:2])
dR = np.linalg.norm(nose_tip[:2] - right_eye_outer[:2])
yaw = (dL - dR) / (dL + dR + 1e-8)

eye_mid = (left_eye_outer + right_eye_outer) / 2
pitch = (nose_tip[1] - eye_mid[1]) / (abs(chin[1] - eye_mid[1]) + 1e-8) - PITCH_CENTRE
```

- `PITCH_CENTRE` = mean of the raw pitch ratio across the Track A training
  set, computed during extraction and shipped in `scaler.json`.
  **Track B uses `0.5` as a placeholder until the real value ships.**
- These are **geometric pose proxies, not calibrated Euler angles**, and are
  described as such everywhere (thesis included). They are monotonic in the
  true angle and scale-invariant, which is all the model needs.

---

## 4. Landmark indices (MediaPipe Face Mesh, 478-point with iris)

**Status: FROZEN — visually verified on Day 2.** Every index was rendered
with its label on a real face (`ml/scripts/verify_landmarks.py`, output in
`docs/verification/`) and 12 automated geometry assertions pass (corner
ordering, brow-above-eye, iris-inside-eye, nose/chin ordering).

```python
LEFT_EYE_EAR  = [33, 160, 158, 133, 153, 144]   # p1..p6
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]  # p1..p6
MOUTH         = [61, 291, 13, 14]               # left, right, upper, lower
LEFT_BROW     = [70, 63, 105, 66, 107]
RIGHT_BROW    = [300, 293, 334, 296, 336]
INTEROCULAR   = [33, 263]                       # outer corners
NOSE_TIP      = 1
CHIN          = 152
LEFT_IRIS     = [468, 469, 470, 471, 472]
RIGHT_IRIS    = [473, 474, 475, 476, 477]
```

The 478-point mesh (with iris points 468–477) is required.

**API decision (Day 1):** mediapipe ≥1.0 removed the legacy
`mp.solutions.face_mesh` API. Both sides therefore use the **Tasks API
FaceLandmarker** — Python: `mediapipe.tasks.python.vision.FaceLandmarker`
(pinned mediapipe 1.0.0), JS: `@mediapipe/tasks-vision`. Both load the
**same `face_landmarker.task` model asset**, which strengthens Python↔JS
landmark parity. Video mode (`running_mode=VIDEO`) in extraction; the
FaceLandmarker outputs all 478 points including iris by default.

### Conventions — DECIDED DAY 2, NOW FROZEN

Documented in capital letters in the `features.py` docstring; repeated here:

1. **Coordinates are PIXELS.** Both sides convert MediaPipe's normalised
   output before calling `compute_features`:
   `x_px = x_norm · frame_width`, `y_px = y_norm · frame_height`,
   `z_px = z_norm · frame_width`. Aspect ratio is thereby handled
   automatically. `frame_shape` is `(HEIGHT, WIDTH)`. z is carried but no
   contract feature currently uses it.
2. **"Left" means IMAGE-LEFT in an un-mirrored frame** — the subject's
   *right* eye/brow for a person facing the camera. Frames must be
   processed un-mirrored on both sides; mirroring is for display only
   (browser selfie views typically mirror via CSS — Track B must feed the
   raw, un-mirrored video frame to the landmarker).
3. **Sign conventions.** y grows downward (image convention). Positive
   `gaze_x` = iris toward image-right; positive `gaze_y` = downward;
   positive `roll` = image-right eye lower than image-left eye; negative
   `yaw` = nose nearer the image-left eye.

---

## 5. Model I/O (FROZEN)

```
INPUT   "features"     [1, 30, 13]  float32   (already standardised with scaler.json)
OUTPUT  "engagement"   [1, 4]       float32   raw logits
OUTPUT  "states"       [1, 4]       float32   raw logits
```

**No softmax or sigmoid inside the graph.** Track B applies them in JS.
Standardisation `(x − mean) / std` happens **outside** the model, in the
caller, using `scaler.json`.

---

## 6. Sampling (FROZEN)

| Parameter | Value |
|---|---|
| Frame rate | 10 FPS |
| Window length | 30 frames (3.0 s) |
| Training stride | 10 frames |
| Inference stride | 5 frames |

> **Amendment 1 — 2026-08-03 (agreed by both partners).** Inference stride
> for the live web app is raised **5 → 30 frames**: one prediction per
> non-overlapping 3.0 s window instead of 2 Hz, for a slower, more readable
> dashboard cadence. Scope: `web/` only (`INFERENCE_STRIDE` in
> `web/hooks/usePipeline.ts`). Frame rate, window length, and training
> stride are unchanged; `ml/` code is untouched. The table above records
> the original Day-1 values.

---

## 7. scaler.json schema (FROZEN)

```json
{
  "mean": [13 floats],
  "std": [13 floats],
  "feature_names": [13 strings, order per Section 2],
  "pitch_centre": 0.5,
  "version": "1.0"
}
```

Fitted on the DAiSEE **training split only**.

---

## 8. Audio / multimodality — ✅ ANSWERED (FROZEN)

> **Audio stream present: NO.** DAiSEE clips contain a single mpeg4 video
> stream (640×480 @ 30 fps, 10 s).
> **Checked:** 2026-08-01 (Day 1–2), via
> `ffprobe -v error -show_streams <clip>` on 9 clips across all three
> splits (e.g. Train 1100011002, Validation 4000221001, Test 5000441001).
> All video-only.

**Therefore, from Day 1 onward, "multimodal" means fusion of three visual
modality families** — geometric (EAR/MAR/brow), pose (yaw/pitch/roll), and
gaze (iris offset). This framing is used consistently in all writing; no
audio features exist anywhere in the project.

Useful extraction facts recorded while checking: 30 fps source → sample
every 3rd frame for the contract's 10 FPS → ~100 feature rows per 10 s
clip. Label CSVs cover 5358 / 1429 / 1784 clips (Train/Val/Test) — fewer
than the ~9,068 on disk. `labels.py` uses the **intersection** of on-disk
clips and label rows, within official split folders only.

---

## 9. Dataset splits (FROZEN)

DAiSEE's **official** Train / Validation / Test folders (5482 / 1723 / 1720
clips) — already subject-independent, directly comparable to the published
benchmark. **No random re-splitting.** Zero clip-ID overlap between splits is
enforced by an `assert` in code, not a comment.

---

---

## Amendment 2 — J1 parity tolerance, and why numFaces:1 feeds the model (2026-08-09)

> **Agreed by both partners, 2026-08-09**, when the J1 gate was rebuilt
> after `web/harness/parity.html` was found to import a file deleted in
> the `cc4f5c8` merge (silently validating code that no longer existed).

**Tolerance: 0.02 max-abs-diff per feature** (after excluding
`face_present` mismatches), not the original 1e-4 from
`BUILD_PLAN_1.md` §J1. Python (`mediapipe.tasks.python`) and the browser
(`@mediapipe/tasks-vision`) run the same `.task` model on the same CPU/
XNNPACK delegate but are different runtimes (native vs WASM); this
produces small, expected sub-pixel landmark noise, not a train/serve bug.
0.02 is the value the team empirically validated: worst-case observed
diff is 0.0079–0.016 depending on harness (`docs/results/parity_report.json`),
comfortably under tolerance, while still tight enough to catch a real
skew (see below).

**GPU delegate excluded, CPU delegate required.** The GPU delegate shifts
landmarks enough to fail even the loosened tolerance —
`docs/results/parity_report_gpu.json` records `gaze_y` at 0.05, over 2×
tolerance. `lib/faceLandmarker.ts` pins `delegate: 'CPU'` for this reason;
this is not optional.

**`numFaces` changes landmark output even for the one real face in
frame.** Rebuilding J1 to test the app's actual production landmarker
config (`numFaces: 4`, added for multi-face overlay support, commit
`c16cd9a`) surfaced a real, previously-undetected regression: on frames
where the subject is blinking, `numFaces: 4` produces a `gaze_y` diff of
up to **0.86** (out of a ~−1..1 range) against the Python reference —
`numFaces: 1` on the identical frames measures 0.016. The old
hand-rolled harness always used its own `numFaces: 1` instance regardless
of what production shipped, so it never could have caught this.

**Resolution:** `lib/faceLandmarker.ts` exports two factories.
`createFeatureLandmarker()` (`numFaces: 1`) is the **only** landmarker
ever allowed to feed `computeFeatures()` — this matches
`ml/src/extract.py`, which always runs `num_faces=1`, so the model never
trained on `numFaces > 1` landmark noise. `createDisplayLandmarker()`
(`numFaces: 4`) drives the on-screen multi-face overlay and "People"
count only, and its output must never reach `computeFeatures()`. Costs a
second WASM detection pass per sampled frame.

## Sign-off

| Partner | Track | Date | Signed |
|---|---|---|---|
| Bilal | A — ML pipeline | 2026-08-03 | ☑ |
| Azeem | B — Web app | 2026-08-03 | ☑ |
| Both | Amendment 2 (J1 rebuild, numFaces fix) | 2026-08-09 | ☑ |
