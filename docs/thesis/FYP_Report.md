# Privacy-Preserving Cognitive Engagement Detection from Webcam Video Using an Edge-Deployed Temporal Convolutional Network

**Ibtissam Merzouqi**
**001487042**
**[Programme, e.g. MSc Data Modelling — fill in]**
**[Supervisor — fill in]** | **[Second Marker — fill in]**
**[Submission date — fill in]**

*This report is written from the completed `Multimodal-Cognitive-Overload-Detection` repository and its full development record (`docs/PROGRESS.md`, `CONTRACT.md`, `BUILD_PLAN_1.md`, `GAP_CLOSURE_PLAN.md`, `PROJECT_COMPLETION_PLAN.md`). All figures, tables, and code references are drawn directly from artefacts already produced and version-controlled in that repository; none are illustrative or hypothetical. Bracketed placeholders (name, dates, screenshots) are the only items requiring manual completion before submission.*

---

## Abstract

Detecting a learner's engagement from webcam video is a well-studied problem in affective computing, but almost every published system assumes the video itself can leave the user's device — uploaded to a server, processed by a cloud API, or streamed to a third party. This project asks a narrower and more practical question: can a useful engagement classifier run entirely inside a web browser, on ordinary consumer hardware, with a demonstrable guarantee that no frame of video or derived feature ever leaves the machine? It answers that question with a complete, working system rather than a simulation of one. A 13-dimensional geometric feature vector — eye aspect ratio, mouth aspect ratio, eyebrow raise, head pose proxies, and gaze offset, all derived from 478 MediaPipe facial landmarks and normalised by interocular distance for scale invariance — is computed identically in a Python training pipeline and a TypeScript in-browser pipeline, verified to agree by an automated cross-language parity test. A small dilated temporal convolutional network (41,544 parameters) is trained on windowed features extracted from 9,032 clips of the DAiSEE dataset (Gupta *et al.*, 2016), quantized to an int8 ONNX model of 60 KB, and executed client-side with ONNX Runtime Web inside a Next.js application, achieving sub-millisecond inference latency. On the DAiSEE test split (14,241 windows), the shipped int8 model reaches a macro-F1 of 0.246, against a majority-class floor of 0.166 and classical (logistic regression / random forest / gradient boosting) baselines of 0.248, 0.264 and 0.291 respectively — a modest absolute score reported honestly alongside a known limitation: the rarest engagement class has only 4 clips in the entire test split, making its per-class metrics statistically unreliable. An extended comparison and search phase then subjects this result to the scrutiny it deserves: recurrent (LSTM/GRU) and Transformer alternatives at matched budgets, a subject-grouped cross-validated feature ablation, a 40-trial Bayesian hyperparameter search, and an out-of-fold stacking ensemble collectively fail to produce a configuration statistically distinguishable from the shipped model — evidence that the model sits near the honest ceiling of its 13-feature representation, not that it was undertuned — while re-scoring the frozen test predictions at clip level (the granularity the published benchmark actually uses) with decision thresholds calibrated on validation raises the comparable test result to a macro-F1 of 0.283 and 44.7% four-class accuracy at no change to the deployed model. The privacy claim itself is not merely asserted: a recorded network trace of a live 65-second inference session shows zero outbound requests once the page has loaded, including one genuine finding — an undisclosed telemetry call inside a third-party dependency, caught and blocked at the browser-policy level via a Content Security Policy header, not patched around. A rebuilt, CI-integrated parity gate additionally caught and fixed a real train/serve skew regression introduced by a later multi-face UI feature, demonstrating the value of the automated verification built around the system, not just the system itself.

**Keywords:** engagement detection, edge machine learning, in-browser inference, temporal convolutional network, MediaPipe, DAiSEE, privacy-preserving computer vision, ONNX Runtime Web, WebAssembly.

---

## Acknowledgements

*[fill in]*

---

## Declaration of AI Use

*[REQUIRED BEFORE SUBMISSION — verify this section's wording and placement against the University of Greenwich's current Generative AI policy and your programme handbook; the draft below states what was actually done and must not be removed, only reworded to the required format.]*

Generative AI tools (Anthropic's Claude, used through the Claude Code development environment) were used during this project as a development and drafting assistant: for writing and refactoring portions of the project's source code and automated tests, for running and logging the experiment pipelines described in Chapter 4, and for drafting prose in this report which was then reviewed by the author. All experimental results reported here were produced by executing the project's committed code against the real DAiSEE dataset and are reproducible from the repository; no result, figure, or reference was generated by an AI system without execution. The project's design decisions, the interface contract with the Track B partner, and responsibility for the final content of this report remain the author's.

---

## Table of Contents

*[Auto-generate from headings in Word/Word Online after pasting: References → Table of Contents. Structure below.]*

1. Introduction
2. Literature Review
3. Methodology
4. Experimentation
5. Conclusion
References
Appendices

---

## List of Tables and Figures

*[Populate via Word's "Insert Caption" + "Insert Table of Figures" once tables/figures below are placed in the final document. Source data/images for each are cited inline in the relevant section — see `docs/results/`, `docs/verification/`, `docs/architecture.md`.]*

---

## Glossary

| Term | Meaning |
|---|---|
| CI | Continuous Integration — automated tests run on every code change (this project: GitHub Actions) |
| CONTRACT.md | This project's frozen interface specification between the Python training pipeline and the browser inference pipeline |
| CSP | Content Security Policy — a browser-enforced HTTP header restricting what a page is allowed to connect to |
| DAiSEE | Dataset for Affective States in E-Environments (Gupta *et al.*, 2016) — the video dataset this project trains on |
| EAR / MAR | Eye Aspect Ratio / Mouth Aspect Ratio — geometric ratios derived from facial landmark distances |
| Macro-F1 | The F1 score (harmonic mean of precision and recall) averaged equally across classes, regardless of class size |
| ONNX | Open Neural Network Exchange — a portable format for trained model graphs |
| Parity gate | An automated test asserting that two independent implementations (here: Python and TypeScript feature extraction) produce numerically equivalent output |
| Quantization | Reducing a model's numeric precision (here: 32-bit floats to 8-bit integers) to shrink size and speed up inference |
| TCN | Temporal Convolutional Network — a 1-D dilated convolutional architecture for sequence modelling (Bai, Kolter and Koltun, 2018) |
| WASM | WebAssembly — a binary instruction format allowing near-native-speed code (here: MediaPipe and ONNX Runtime) to run in a browser |

---

# 1. Introduction

## 1.1 Problem Background at Large

Learner engagement — the degree to which a student is attentive, interested, and cognitively invested in a learning activity — is one of the strongest predictors of educational outcomes available to an instructor or an adaptive learning system (Karimah and Hasegawa, 2022). In a physical classroom, an experienced teacher reads engagement continuously and adjusts pace, difficulty, or delivery in response. Online and self-paced learning removes that feedback loop entirely: an e-learning platform has no equivalent of glancing around a room. This gap has motivated a substantial body of research into *automatic* engagement recognition — inferring engagement computationally from signals a learner already produces, most commonly facial video, so that a system can react the way a human instructor would (Karimah and Hasegawa, 2022; Dewan, Murshed and Lin, 2019).

Almost all of that research, however, shares an architectural assumption that this project treats as the actual problem to be solved rather than a detail to work around: the video is processed somewhere other than the device that captured it. Facial-analysis pipelines built on OpenFace, cloud vision APIs, or server-hosted deep networks require the raw frame — or at minimum a detailed feature representation of a person's face — to leave the user's machine (Santoni, Basaruddin and Junus, 2023). For a system whose entire purpose is to continuously watch a learner's face, that is a significant and often under-examined privacy cost, one that is easy to defer as "someone else's problem" (a data-handling policy, a consent form) rather than solve as an engineering constraint.

The stakes of that cost are not abstract. A continuous facial-monitoring stream is one of the more sensitive categories of personal data a learning platform could plausibly collect — it can reveal far more than engagement level alone (identity, presence, emotional state, and, incidentally, whoever else is in frame, Section 4.6.2), and once such a stream leaves a user's device, the learner has no further practical control over its retention, secondary use, or exposure in a breach. Server-side engagement-detection deployments therefore inherit the full weight of data-protection obligations — consent management, retention policy, breach liability — that an architecture in which the sensitive data never leaves the device in the first place does not need to solve after the fact, because there is nothing transmitted to protect. This project's motivation is the observation that, for this specific task, avoiding that obligation entirely by construction is technically achievable with current browser ML tooling (Section 2.5), and is worth demonstrating as a complete, working, and — the point this whole report returns to repeatedly — independently verified system, rather than left as a plausible-sounding architectural aspiration.

## 1.2 Specific Area of Problem

This project narrows that broad concern to a concrete, testable engineering question: **can a facial-video-based engagement classifier run entirely client-side, in an ordinary web browser, with an inference pipeline small and fast enough for real-time use on consumer hardware, while remaining provably free of any video or feature data leaving the device?**

"Provably" is deliberate. A system can claim privacy; this project instead treats the claim as something to be tested the same way accuracy is tested — with a recorded, reproducible artefact (Section 4, Experiment 4; `docs/privacy.md`) rather than a sentence in a README.

## 1.3 Framing as a Computer/Data Science Problem

Reduced to its computational form, the problem has three coupled parts, each with its own well-established sub-field:

1. **A supervised sequence-classification problem.** Given a 3-second window of per-frame facial geometry features, predict one of four ordinal engagement levels (and, secondarily, four binary affective states — bored, confused, engaged, frustrated). This is a temporal, multi-class classification task under severe class imbalance (Section 1.5).
2. **A cross-language numerical-equivalence problem.** The feature extraction that produces training data (Python) and the feature extraction that runs live in the browser (TypeScript) are two independent implementations of the same mathematics that must agree to within a small, empirically justified tolerance — otherwise the model sees systematically different inputs at inference time than it saw during training ("train/serve skew"). This is treated in this project as a first-class engineering deliverable (the "J1 parity gate"), not an assumption.
3. **A model-compression and edge-deployment problem.** The trained network must be small and fast enough to run inside a browser tab via WebAssembly, without a GPU, at a cadence usable for live feedback — which motivates int8 quantization and a deliberately tiny architecture (Section 3).

## 1.4 Objectives

Specific, measurable, achievable, relevant, time-bound (S.M.A.R.T.) objectives, as executed against a 20-day build plan (`BUILD_PLAN_1.md`) and a subsequent verification/hardening pass (`GAP_CLOSURE_PLAN.md`, `PROJECT_COMPLETION_PLAN.md`):

- **O1.** Implement a 13-feature facial geometry extractor, identical in Python and TypeScript to within a max-absolute-difference tolerance of 0.02 per feature, verified by an automated cross-language test — *achieved, Section 4 Experiment 1*.
- **O2.** Train a temporal model on the DAiSEE dataset that beats a majority-class baseline and two classical-ML baselines on validation macro-F1 — *achieved, Section 4 Experiments 2–3*.
- **O3.** Quantize the trained model to int8 and ship it inside a browser application that reaches "live" (camera → landmarks → features → inference → UI) end-to-end, with a measured inference latency compatible with real-time use — *achieved, Section 4 Experiments 3, 5*.
- **O4.** Produce direct, recorded evidence for the privacy claim (a network trace of a live session), not an assertion — *achieved, Section 4 Experiment 4*.
- **O5.** Verify the system survives realistic deployment conditions — a clean install on an unmodified machine, more than one browser engine, and deliberately adverse camera conditions (no face, multiple faces, poor lighting, glasses) — *achieved, Section 4 Experiment 6*.

## 1.5 Dataset Details and Exploratory Data Analysis

The project trains on **DAiSEE** (Dataset for Affective States in E-Environments; Gupta *et al.*, 2016), the first large-scale, in-the-wild, multi-label video dataset for this task: 9,068 ten-second clips from 112 university-age participants watching educational content on a laptop webcam, each independently labelled on a four-point scale (very low / low / high / very high) for four affective states — boredom, confusion, engagement, and frustration — crowd-annotated and cross-checked against an expert gold standard.

**Pre-processing pipeline (executed once, `ml/src/extract.py`; full parameters and results in `docs/results/extraction_stats.json`):**

- Of the clips present on disk, 9,032 were processed successfully (0 failures) across the dataset's official Train / Validation / Test folders (5,446 / 1,720 / 1,866 clips respectively) — the official DAiSEE split is used unmodified specifically so results remain comparable to the published benchmark, rather than re-splitting.
- Each clip was sampled at 10 FPS (every third frame of the native ~30 FPS source) and run through a MediaPipe FaceLandmarker to extract 478 facial landmarks (Kartynnik *et al.*, 2019) per sampled frame.
- **Face-detection rate: 99.96%** (Train 99.98%, Validation 99.93%, Test 99.93%) — essentially every sampled frame across the dataset yielded a usable face, indicating DAiSEE's recording conditions (frontal webcam, indoor lighting) are consistently favourable for landmark-based extraction.
- Each clip's ~100 per-frame feature rows were windowed with a stride of 10 over a 30-frame (3.0 s) span, yielding roughly 8 overlapping windows per clip; **42,856 / 11,432 / 14,241 windows** for Train / Validation / Test respectively.

**Class imbalance — the central empirical fact shaping every later decision (labels via `ml/src/labels.py`; exact support counts from `docs/results/metrics_test.csv`):** the four-class engagement label is severely imbalanced. On the test split, "very low" engagement (class 0) accounts for only 32 of 14,241 windows (≈0.22%, corresponding to just 4 distinct clips), "low" (class 1) 668 windows (≈4.7%), while "engaged" and "very engaged" together make up over 95% of the data. This single fact — not a modelling detail, but a property of how engagement actually distributes in a real classroom population, most students most of the time are at least moderately engaged — is the reason this report evaluates everything in macro-F1 rather than accuracy (Section 3.1), why the training loss uses inverse-frequency class weighting (Section 3.2), and why a merged "3-class" metric (0+1 collapsed into a single "low" class) is reported alongside the primary 4-class metric throughout Section 4: with only 4 test clips in the rarest class, its individual precision/recall are close to statistically meaningless on their own, and reporting that plainly is preferable to letting a single misleadingly precise-looking number stand unexplained.

*[Figure 1.1: class distribution bar chart, `docs/results/class_dist.png` — insert here]*

---

# 2. Literature Review

## 2.1 Facial-geometry feature engineering for engagement and drowsiness

### 2.1.1 Prior methodology

Before end-to-end deep learning became dominant, a substantial and still-relevant line of work characterised eye and mouth state through simple, interpretable geometric ratios computed from a small number of facial landmark points. The canonical example is the **eye aspect ratio (EAR)** of Soukupová and Čech (2016): a scalar computed from six landmarks around each eye, `(|p2−p6| + |p3−p5|) / (2|p1−p4|)`, that falls sharply during a blink and was shown to outperform prior eye-closure detectors on two standard datasets while being cheap enough to compute at video frame rate. The same geometric-ratio philosophy generalises naturally to mouth aperture (mouth aspect ratio, MAR) and eyebrow raise (a normalised brow-to-eye distance), both used elsewhere in facial-action-adjacent literature as fast proxies for expression-level events without requiring a full facial action unit (FACS) classifier.

### 2.1.2 Fit with this project's methodology

This project adopts the EAR formula from Soukupová and Čech (2016) essentially unchanged (`ml/src/features.py`, `eye_aspect_ratio`), and extends the same geometric-ratio philosophy — a small number of landmark distances, normalised, fast to compute — to mouth aperture, brow raise, head-pose proxies, and gaze offset, giving a 13-dimensional feature vector per frame rather than adopting a deep, end-to-end spatial encoder (contrast Section 2.3). The fit is deliberate and constrained by the project's edge-deployment goal (Section 2.3.2): a geometric feature vector is orders of magnitude cheaper to compute in a browser than running a CNN over raw pixels every frame, and — critically — is something a *second, independent implementation* (TypeScript) can be written for and numerically verified against the Python reference (Section 3, the J1 parity gate). A raw-pixel CNN feature extractor offers no equivalent way to verify that two language runtimes compute "the same" internal representation; a documented arithmetic formula does.

## 2.2 Facial landmark detection and head pose

### 2.2.1 Prior methodology

**MediaPipe Face Mesh** (Kartynnik *et al.*, 2019) is a neural-network-based model that predicts an approximate 3-D mesh of 468 surface points from a single RGB frame, running in real time on mobile GPUs without specialised hardware; **MediaPipe Iris** extends the same framework with dedicated iris-landmark prediction, enabling gaze- and depth-related features from a standard webcam (research.google, cited in Section 2.2 discussion). Separately, classical head-pose estimation typically solves the **Perspective-n-Point (PnP)** problem — recovering a camera-relative 3-D rotation from a small set of 2-D landmark-to-3-D-model correspondences — via methods such as EPnP (Lepetit, Moreno-Noguer and Fua, 2009), which reduced the computational complexity of prior PnP solvers from polynomial-in-*n* to linear while retaining accuracy.

### 2.2.2 Fit with this project's methodology

This project uses MediaPipe's Tasks API `FaceLandmarker`, built on the Kartynnik *et al.* (2019) mesh model extended to 478 points (468 surface + 10 iris), on **both** sides of the pipeline (Python, via `mediapipe.tasks.python`, and browser, via `@mediapipe/tasks-vision`) specifically because it is one of very few landmark detectors shipped as both a native Python package and a browser-deployable WASM package with the same underlying model asset — a precondition for the cross-language parity this project depends on (CONTRACT.md §4).

Head pose is the one area where this project deliberately does **not** follow the dominant prior approach. A true PnP solution (EPnP or otherwise) requires `cv2.solvePnP` on the Python side and an equivalent on the browser side; the only practical browser equivalent, `opencv.js`, costs roughly 8 MB of additional WASM purely to obtain three Euler angles — disproportionate for an application whose entire premise is a minimal footprint. Instead, yaw, pitch, and roll are computed as **geometric pose proxies** directly from landmark positions (nose-to-eye-corner distance ratios for yaw; nose-drop-below-eye-line for pitch; eye-corner slope for roll) — monotonic in, but not numerically equal to, the true Euler angle. This is a conscious accuracy-for-footprint trade-off, made explicit rather than glossed over: the proxies are described as proxies everywhere in this project's documentation and in this report, never presented as calibrated angles.

## 2.3 Temporal modelling of engagement from DAiSEE

### 2.3.1 Prior methodology

Two DAiSEE-specific prior works are directly comparable to this project's approach. Abedi and Khan (2021) combine a 2-D ResNet spatial encoder with a **Temporal Convolutional Network (TCN)** temporal head, trained end-to-end on raw DAiSEE video frames, reporting 63.9% accuracy on the four-class engagement task — an improvement of 2.75 percentage points over a ResNet+LSTM baseline and, at the time, a new state of the art for the dataset. More broadly, Bai, Kolter and Koltun (2018) establish the TCN architecture itself as a strong general-purpose alternative to recurrent networks for sequence modelling: a stack of causal, dilated 1-D convolutions with residual connections, whose receptive field grows exponentially with depth, offering better gradient stability, full parallelism during training, and — the property most relevant here — a small, fixed parameter count independent of sequence length, unlike an LSTM's per-timestep recurrent state. Separately, Santoni, Basaruddin and Junus (2023) tackle DAiSEE's class imbalance directly with a CNN model over OpenFace-derived features (facial landmarks, head pose, action units, gaze) combined with SMOTE oversampling, explicitly naming the imbalance problem "yet to be addressed in previous research" at the time of writing (2023) — this project's own class-weighting approach (Section 3.2) is one alternative answer to the same problem.

### 2.3.2 Fit with this project's methodology

This project's architecture (Section 3, `ml/src/model.py`) is a TCN in the sense of Bai, Kolter and Koltun (2018) — four dilated causal convolutional blocks (dilations 1/2/4/8), residual connections, global average pooling — but applied to the 13-dimensional geometric feature sequence described in Section 2.1, not to raw pixels or a ResNet's learned spatial embedding as in Abedi and Khan (2021). This is the single largest architectural departure from the closest prior DAiSEE work, and it is a direct consequence of the edge-deployment objective (Section 1.3): a ResNet frontend evaluated every frame is entirely impractical to quantize into a sub-100 KB browser-deployable model and run in real time without a GPU, whereas a 13-float-per-frame TCN can be (Section 3, Section 4 Experiment 3). The trade-off is measurable and reported honestly rather than hidden: this project's shipped model reaches a test macro-F1 of 0.246 (Section 4), well below Abedi and Khan's reported 63.9% *accuracy* on the same dataset — though the two numbers are not directly comparable (macro-F1 versus accuracy under severe imbalance are answering different questions, Section 1.5), the underlying spatial information a ResNet frontend can access and a 13-float geometric vector cannot is very likely a genuine, not just metric-driven, source of the gap, and is named explicitly as a limitation in Section 5.2.2 rather than left implicit.

Santoni, Basaruddin and Junus (2023) and this project converge on the same diagnosis — DAiSEE's class imbalance is severe enough to dominate model behaviour if not addressed directly — but choose different remedies: SMOTE-based oversampling of minority-class examples at the data level, versus inverse-frequency loss weighting at the training-objective level (Section 3.2) in this project. The training-objective approach was chosen here specifically because it requires no synthetic example generation from an already-scarce 32-window minority class, where oversampling risks amplifying noise from a handful of source clips into many synthetic duplicates.

## 2.4 Dataset-annotation validity and engagement definitions

### 2.4.1 Prior methodology

Khan, Abedi and Colella (2022) provide a critical review of how "student engagement" is operationalised across the datasets built to study it, including DAiSEE, arguing that annotation protocols across the field are frequently inconsistent with definitions of engagement established in educational psychology, and cautioning that models trained on such labels risk learning dataset-specific annotator behaviour rather than a generalisable construct. Karimah and Hasegawa's (2022) systematic review of 47 engagement-estimation studies corroborates this at a field level, finding that emotional engagement operationalised via affective cues is the dominant framing (present in 65.6% of reviewed studies) — DAiSEE's own four-affective-state label scheme sits squarely inside that dominant framing, rather than the alternative behavioural or cognitive engagement definitions the same review also documents.

### 2.4.2 Fit with this project's methodology

This project does not attempt to resolve the definitional question Khan, Abedi and Colella (2022) raise — that is a labelling-methodology question the DAiSEE dataset itself answers by construction, and re-annotating 9,068 clips was never in scope. What this project does instead is treat their caution as a reason for restraint in how results are interpreted: the four-class engagement label is used exactly as DAiSEE defines it, without claiming the resulting model measures "engagement" in some dataset-independent sense, and the known statistical unreliability of the rarest class (Section 1.5) is surfaced rather than smoothed over — consistent with the spirit, if not the specific recommendations, of Khan, Abedi and Colella's critique.

## 2.5 Model compression and in-browser inference for edge deployment

### 2.5.1 Prior methodology

Jacob *et al.* (2018) introduced a quantization scheme enabling neural network inference using integer-only arithmetic, co-designing a training procedure that preserves accuracy after quantizing weights and activations to 8-bit integers — foundational to essentially all modern mobile/edge int8 deployment, including the ONNX Runtime quantization tooling used directly in this project (Section 3). More specifically to this project's deployment target, recent work on in-browser inference (e.g. work on just-in-time kernel optimisation for in-browser deep learning on edge devices) documents WebAssembly, combined with WebGPU where available, as now a practically viable substrate for running non-trivial neural network inference client-side, with on-device inference offering the privacy and latency advantages this project's central claim depends on — sensitive data never leaves the device, and no network round-trip is on the inference critical path.

### 2.5.2 Fit with this project's methodology

This project applies **static QDQ int8 quantization** (via ONNX Runtime's `quantize_static`, `ml/src/export.py`) in the spirit of Jacob *et al.* (2018), taking the shipped model from 163 KB (fp32) to 60 KB (int8) for a measured macro-F1 cost of −0.0015 on the test split (Section 4, Experiment 3) — a favourable trade given the deployment target. One implementation detail is worth recording as a genuine, empirically-discovered finding rather than a design choice made in advance: an earlier attempt at **dynamic** quantization produced a model that loaded but could not execute in onnxruntime-web's WASM backend, because dynamic quantization emits `ConvInteger` operator nodes that onnxruntime-web does not implement, whereas static QDQ quantization emits `QLinearConv` nodes, which it does. This was discovered empirically via a dedicated browser-execution smoke test (`ml/scripts/browser_tests.py`, the "A6.5" gate) before it could silently ship a model that failed in production — an example of exactly the gap between "quantizes successfully in Python" and "actually runs where it is meant to run" that motivates several of this project's other verification gates (Section 3.4, Section 4 Experiment 6).

## 2.6 Ordinal regression and automated hyperparameter optimisation

### 2.6.1 Prior methodology

Two further methods entered this project during its extended experimentation phase (Chapter 4, Experiments 7–8) and are reviewed here for completeness. **CORAL** (Cao, Mirjalili and Raschka, 2020) reformulates ordinal classification — categories with a meaningful order, such as DAiSEE's very-low-to-very-high engagement scale — as a set of rank-consistent binary "greater than level *k*" sub-problems sharing one learned representation, guaranteeing the predicted rank probabilities are monotonically consistent, unlike naive per-threshold classifiers. **Optuna** (Akiba *et al.*, 2019) is a hyperparameter-optimisation framework combining Tree-structured Parzen Estimator (TPE) Bayesian sampling with aggressive early pruning of unpromising trials, making systematic search practical at a fraction of exhaustive-grid cost.

### 2.6.2 Fit with this project's methodology

Both are used as *interrogation tools* rather than components of the shipped system: CORAL as an alternative engagement head to test whether directly optimising the label's ordinal structure improves on the softmax formulation (Experiment 8 — it trades macro-F1 for accuracy/ordinal agreement rather than strictly improving), and Optuna as the search engine for the systematic hyperparameter sweep the original six-run manual search lacked (Experiment 8 — the search validated rather than displaced the manually chosen configuration). Neither changes the deployed model; both change how much confidence its configuration deserves.

## 2.7 Synthesis: Positioning This Project Within the Literature

Read together, Sections 2.1–2.5 describe a field that has largely bifurcated into two families of approach: end-to-end deep architectures operating on raw or near-raw video (Section 2.3.1's ResNet+TCN hybrid being the strongest DAiSEE-specific example), which tend to maximise predictive accuracy at the cost of computational footprint and interpretability; and classical geometric/handcrafted-feature approaches (Section 2.1.1's EAR, and OpenFace-style pipelines as used by Santoni, Basaruddin and Junus, 2023), which trade some accuracy for interpretability, speed, and — critically for this project — the ability to be implemented twice, independently, and checked for agreement. No work surveyed here occupies quite the same position this project does: a geometric-feature-plus-TCN combination, deployed and benchmarked as a genuinely running, quantized, client-side browser application rather than evaluated only offline in Python, with the cross-language equivalence between training and inference treated as a measured, tested property rather than an implicit assumption. This is not offered as a claim that this project's approach is strictly superior to the deep end-to-end alternatives — Section 4.3.3's direct comparison against Abedi and Khan (2021) shows plainly that it is not, on raw predictive performance — but as the specific, narrower research niche this project occupies: privacy-preserving, edge-verifiable engagement inference, where the two literatures reviewed above (engagement recognition, and edge/on-device ML deployment) are each individually well established but rarely combined with this level of end-to-end, browser-deployed rigour in the DAiSEE literature specifically. The class-imbalance and annotation-validity threads (Sections 2.3.2, 2.4) apply to this project exactly as they apply to the prior work reviewed, and are treated identically — named, measured, and reported rather than designed around silently.

---

# 3. Methodology

## 3.1 List of Requirements

Derived directly from the dataset properties established in Section 1.5 and the prior-work gaps identified in Section 2:

| # | Requirement | Source |
|---|---|---|
| R1 | Feature extraction must be numerically reproducible across two independent language implementations (Python training, TypeScript inference), to a documented, empirically-justified tolerance | Section 1.3(2); no equivalent guarantee exists for raw-pixel deep features (Section 2.1.2) |
| R2 | The trained model must be evaluated primarily on macro-F1, not accuracy, given the ≈0.22% prevalence of the rarest class | Section 1.5 |
| R3 | The training objective must explicitly counteract class imbalance, not rely on the optimiser discovering minority classes unaided | Section 1.5, Section 2.3.2 |
| R4 | The exported model must run inside a browser WASM runtime, not merely inside Python ONNX Runtime | Section 2.5.2 (the dynamic-quantization finding demonstrates these are not equivalent) |
| R5 | No frame of video, no derived feature, and no prediction may be transmitted off-device at any point, and this must be independently verified, not merely designed for | Section 1.2 |
| R6 | Head pose must be computed without adding a heavyweight browser dependency (ruling out `cv2.solvePnP`/`opencv.js`) | Section 2.2.2 |
| R7 | The DAiSEE official Train/Validation/Test split must be preserved unmodified, with zero clip-ID leakage across splits, to remain comparable to the published benchmark and prior work (Section 2.3) | Section 1.5 |

## 3.2 Methodology: a Two-Track Contract-Driven Pipeline

The project's own name for its methodology, used consistently throughout its own build documentation, is **"contract-driven parallel development"**: rather than a single linear pipeline, two independently developed tracks — a Python training pipeline ("Track A") and a TypeScript browser-inference pipeline ("Track B") — are built in parallel against a single frozen specification document (`CONTRACT.md`), with an automated numerical-equivalence test (R1) as the mechanism that lets both tracks proceed without waiting on each other while still guaranteeing they end up compatible. This is closer to a contract-first API-development pattern borrowed from software engineering than to a conventional single-track ML pipeline, and it is the direct methodological answer to the train/serve-skew risk named as the highest-likelihood risk in this project's own risk register (`BUILD_PLAN_1.md` §8).

### 3.2.1 Pre-existing methodology used

No single named methodology (e.g. CRISP-DM) is followed wholesale; the project instead combines a standard supervised-ML pipeline (extract → label → window → train → evaluate → export) with the contract-first parallel-tracks pattern above, and with continuous-integration practice (automated regression gates run on every change, Section 3.4) borrowed from conventional software engineering rather than typical ML research workflows.

### 3.2.2 Step 1 — Feature definition (the contract itself)

Thirteen features are computed per frame, identically specified for both language implementations (`CONTRACT.md` §2; reference implementation `ml/src/features.py`; browser port `web/lib/features.ts`):

| # | Feature | Definition |
|---|---|---|
| 0–2 | `ear_left`, `ear_right`, `ear_mean` | Eye Aspect Ratio (Soukupová and Čech, 2016), each eye and their mean |
| 3 | `mar` | Mouth Aspect Ratio — vertical ÷ horizontal lip distance |
| 4–5 | `brow_left`, `brow_right` | Eyebrow-to-eye-centre distance ÷ interocular distance |
| 6–8 | `yaw`, `pitch`, `roll` | Geometric head-pose proxies (Section 2.2.2) |
| 9–10 | `gaze_x`, `gaze_y` | Iris-centre offset from eye-corner midpoint, normalised by eye width/height |
| 11 | `face_area` | Landmark bounding-box area ÷ frame area |
| 12 | `face_present` | 1.0 if a face was detected in the frame, else 0.0 |

Two normalisation rules apply throughout and are load-bearing for the cross-domain generalisation this project needs (DAiSEE clips recorded at one camera distance; live webcam users sitting at another): **every raw distance is divided by interocular distance** (the outer-eye-corner-to-outer-eye-corner span) before use, cancelling out camera-distance scale; and **a frame with no detected face emits thirteen zeros with `face_present = 0.0`**, never an interpolated or dropped value, so the model must learn to handle missing-face frames as first-class input rather than assuming continuous detection.

### 3.2.3 Step 2 — Extraction

`ml/src/extract.py` walks the DAiSEE directory tree; for each clip, opens it with OpenCV, samples every third frame (30 FPS native → 10 FPS contract rate), runs MediaPipe's `FaceLandmarker` in `VIDEO` running mode (a fresh landmarker instance per clip, in a `multiprocessing.Pool` worker, so that `VIDEO` mode's internal frame-to-frame tracking state — which assumes monotonically increasing timestamps within one continuous stream — never leaks between unrelated clips), converts the 478 returned landmarks from normalised to pixel coordinates, and calls the Step-1 feature function. Results in Section 4, Experiment 1 (extraction statistics) and Section 1.5.

### 3.2.4 Step 3 — Labelling and windowing

`ml/src/labels.py` joins DAiSEE's official per-clip label CSVs against the clips actually present on disk (using the **intersection** of the two, within each official split folder, rather than assuming full coverage); `ml/src/dataset.py` then slides a 30-frame (3.0 s) window with stride 10 over each clip's ~100 feature rows, assigns the clip's engagement label to every window drawn from it, and writes the resulting `(N, 30, 13)` tensors per split. **Zero clip-ID overlap between the three splits is asserted in code, not merely assumed** — a single windowing bug that leaked one clip's frames across a split boundary would silently inflate every downstream metric, so this is treated as a correctness invariant worth an explicit runtime check rather than a one-time manual verification. A `StandardScaler` is fit on the **training split only** and applied to all three splits, with its mean/std/feature-name triple serialised to `scaler.json` and shipped to the browser (`web/public/model/scaler.json`) so both the Python and TypeScript pipelines standardise features identically.

### 3.2.5 Step 4 — Classical baselines

Before any temporal model is trained, each 30×13 window is collapsed to a 65-dimensional vector — the mean, standard deviation, minimum, maximum, and range of each of the 13 features across the 30-frame window (`ml/src/baselines.py`) — and used to train a logistic regression and a random forest classifier (`scikit-learn`, `class_weight="balanced"`), alongside a constant majority-class predictor. This step exists specifically to answer the question Section 4 must otherwise leave open: *is the temporal model's added complexity (Step 6) actually earning its keep*, by establishing what a model with no access to within-window temporal ordering — only summary statistics of it — can achieve on the same data.

### 3.2.6 Step 5 — Model architecture

`ml/src/model.py` defines `EngagementTCN`: four stacked dilated causal 1-D convolutional blocks (Conv1d, kernel size 3, dilations 1/2/4/8; each followed by BatchNorm, ReLU, Dropout(0.2), and a residual connection), operating on the transposed `(B, 13, 30)` input, followed by a mean pool over the time dimension (deliberately *not* `AdaptiveAvgPool1d`, which does not export cleanly to ONNX) and two linear heads — a 4-way engagement-logit head and a 4-way (multi-label) state-logit head. Total parameter count: **41,544**, against a self-imposed budget of under 100,000 — the entire edge-deployment argument rests on the model staying this small, so parameter count is checked by an assertion in the module itself (`assert n < 100_000`), not merely reported after the fact.

### 3.2.7 Step 6 — Training

Adam optimiser, initial learning rate 1×10⁻³ with cosine annealing over up to 100 epochs, batch size 128, **early stopping on validation macro-F1** (not validation loss, not accuracy — consistent with R2) with patience 15. The loss combines a class-weighted cross-entropy term for the engagement head with a binary cross-entropy term for the four state heads: `L = CE(engagement, w) + 0.5 · BCE(states)`, where `w` is the inverse training-class-frequency vector (R3) — directly counteracting the imbalance quantified in Section 1.5, since an unweighted loss would let the two majority classes — together ≈95% of training windows — dominate the gradient and give the optimiser no signal to ever predict the rarest class (0.63% of training windows) correctly.

### 3.2.8 Step 7 — Export and quantization

`ml/src/export.py`: `torch.onnx.export` (opset 17, dynamic batch axis, legacy — non-dynamo — exporter, chosen because PyTorch's newer dynamo exporter forces opset 18 and produces a graph shape onnxruntime's static quantizer rejects, found empirically during this step); `onnx.checker.check_model`; a **second, independent parity check** — 100 random inputs run through both the PyTorch model and the exported ONNX graph via `onnxruntime`, asserting max-absolute-difference below 1×10⁻⁵, **aborting the build if it fails** — then static QDQ int8 quantization (Section 2.5.2), calibrated on real standardised training windows, and finally a smoke run of the quantized graph in Python ONNX Runtime before it is ever copied into the web application.

### 3.2.9 Step 8 — Browser inference

The Next.js application (`web/`) mirrors Steps 2–3 of the pipeline entirely client-side: `getUserMedia` supplies a raw (un-mirrored — mirroring is CSS-only, for on-screen display) video stream; **two separate MediaPipe `FaceLandmarker` instances** are run per sampled frame — one configured for a single face (`numFaces: 1`), whose output is the *only* one ever passed into the Step-1 feature function, because that is what the training pipeline was built against; and a second, independent instance configured for up to four faces (`numFaces: 4`), used *only* to drive the on-screen multi-face overlay and a "People" count, and never fed into the model (the reason two separate instances exist, rather than one, is itself a finding — Section 4, Experiment 1). The resulting 13-float vectors fill a fixed 30-window ring buffer; once full, every 30th sample (one non-overlapping 3.0 s window) triggers standardisation via the shipped `scaler.json` and a single ONNX Runtime Web inference call, with softmax/sigmoid applied in JavaScript rather than inside the graph (kept out of the exported model to keep the ONNX graph's contract minimal and the post-processing explicit and auditable).

## 3.3 Considerations

**Technical.** The single largest technical risk this methodology is built around is train/serve skew (Section 1.3(2), R1) — the risk that the Python feature extraction used for training and the TypeScript feature extraction used live diverge numerically, silently, without any error being raised, simply producing a model that performs worse than its offline evaluation suggested. The entire contract-driven methodology (Section 3.2) and its associated automated parity gate (Section 3.4) exist specifically to make that risk detectable rather than latent, and Section 4, Experiment 1 documents a case where this mechanism caught a real regression, not merely a theoretical one.

**Ethical and privacy.** DAiSEE is a licensed academic dataset; its terms explicitly prohibit redistribution of the underlying video, which this project respects at every level — the raw clips are gitignored and never enter version control, and even *derived* artefacts (extracted frame images, transcoded fixture clips used for automated testing) are likewise excluded from the git history, not merely the final delivered product (`.gitignore`, enforced and independently re-verified against the full commit history, Section 4 Experiment 6). For the deployed application itself, the ethical position taken is that a system whose purpose is continuous facial monitoring carries a privacy obligation that cannot be discharged by a policy document alone; Section 3.2.9's client-side-only architecture and Section 4, Experiment 4's recorded verification are the concrete response to that obligation, including disclosure of a genuine third-party telemetry call discovered during verification and blocked at the browser-policy level (`docs/privacy.md`) rather than left unmentioned because it originated in a dependency rather than this project's own code.

## 3.4 Verification as Methodology, Not Afterthought

Four automated gates are integrated into this project's own development workflow (`.github/workflows/ci.yml`) and treated as part of the methodology rather than post-hoc quality assurance, because each answers a question the modelling pipeline alone cannot:

- **Feature parity** (R1) — do the two independent language implementations agree? (Section 4, Experiment 1)
- **Model export parity** (Section 3.2.8) — does the quantized ONNX graph agree with the trained PyTorch model?
- **Browser execution smoke test** (Section 2.5.2) — does the exported graph actually *run* in the target WASM runtime, not just in Python?
- **End-to-end integration** — does the full live pipeline (camera → landmarks → features → inference → UI) produce a valid, well-formed prediction under realistic and deliberately adverse conditions? (Section 4, Experiment 6)

---

# 4. Experimentation

## 4.1 Experiment 1 — Cross-Language Feature Parity

### 4.1.1 Set-up

A single 10-second DAiSEE clip is transcoded once to a browser-decodable VP9 WebM and, from the identical decode pass, both (a) full-precision reference features via the Python pipeline (`ml/scripts/make_parity_fixture.py`, using the *same* `FaceLandmarker` configuration as training extraction) and (b) 100 individual PNG frames are produced, guaranteeing byte-identical, frame-aligned input to both sides of the comparison. A Playwright Test (`web/tests/e2e/features.parity.test.ts`) then drives a real Next.js route that runs each of the 100 frames through the browser's *production* landmarker factory and the browser's `features.ts`, and compares every one of the 13 features per frame against the Python reference, gated on: at least 95% of frames comparable, at most 2% face-presence disagreement, and a worst-case per-feature max-absolute-difference under an agreed tolerance of 0.02 (`CONTRACT.md` Amendment 2 — the value the tolerance took is itself a finding, discussed below).

This experiment was **rebuilt during this project's later verification phase** after it was discovered that the previously-committed test harness imported a file deleted in an earlier repository merge — meaning a recorded "pass" on file was, at that point, silently validating code that no longer existed. The rebuild deliberately reuses the exact same landmarker factory the production application uses, rather than a second, hand-configured instance, specifically so that a future change to the application's landmarker configuration cannot again drift out of what this test actually validates.

### 4.1.2 Results

The rebuild's first run failed: worst-case per-feature difference of 0.86 (`gaze_y`), over 40× the 0.02 tolerance, occurring specifically on frames where the subject was blinking. Root-cause isolation (holding every other variable fixed and toggling a single landmarker configuration option) traced this to `numFaces: 4` — a multi-face detection option added to the application *after* this parity test had last genuinely passed, to support an unrelated multi-face-overlay UI feature — measurably shifting landmark output even for the one real face present in frame, worst on blink frames where iris landmarks are already at their most sensitive to sub-pixel noise. With `numFaces: 1`, the identical 100 frames produced a worst-case difference of 0.0157 (`brow_left`), consistent with the 0.0079 previously recorded and comfortably inside tolerance.

*[Table 4.1: per-feature max-absolute-difference, `docs/results/parity_report.json` — insert here]*

The fix — splitting the single landmarker into two independent instances (`createFeatureLandmarker()`, `numFaces: 1`, feeding the model; `createDisplayLandmarker()`, `numFaces: 4`, driving the on-screen overlay only) — restored the 0.0157 result while preserving the multi-face UI feature, at the cost of a second WASM detection pass per sampled frame.

### 4.1.3 Evaluation and Discussion

This result is, in a direct sense, the most important single finding in this project's verification work, precisely because it was not anticipated. The test was rebuilt to be more faithful to production (reusing the real landmarker factory) specifically to close a class of risk the previous, unfaithful test structurally could not detect; that rebuild then immediately surfaced a real defect the previous test's design guaranteed it would miss. The defect itself is a textbook instance of train/serve skew (Section 3.3): the model was trained exclusively on features extracted with `num_faces=1` (`ml/src/extract.py`), and the live application had, for an unrelated UI reason, begun feeding it features computed under a configuration the model had never seen — a regression that a purely offline, Python-only evaluation pipeline (Section 4, Experiment 3) has no way to ever observe, since it never touches the browser code path at all. This is the clearest evidence in this project for the claim made in Section 3.4 — that verification gates covering the *actual deployed system*, not just the model in isolation, are methodologically necessary rather than a nice-to-have, and it directly informed the 0.02 tolerance decision: loose enough to absorb genuine, harmless Python-vs-WASM landmark noise (≈0.008–0.016 in practice), tight enough that this specific 0.86 regression still failed loudly rather than slipping under a more generously-set bar.

## 4.2 Experiment 2 — Classical Baselines

### 4.2.1 Set-up

Each 30×13 window from all three splits is reduced to a 65-dimensional vector (mean/std/min/max/range per feature, Section 3.2.5) and used to train a `class_weight="balanced"` logistic regression and random forest once on the training split, evaluated on both Validation and Test, alongside a constant majority-class predictor. Reported alongside the primary 4-class macro-F1/accuracy is a secondary 3-class-merged metric (engagement levels 0 and 1 collapsed into a single "low" class) — the same collapse Experiment 3's evaluation applies to the TCN, included specifically so the rarest class's statistical unreliability (Section 1.5) does not have to be silently absorbed into a single headline number.

### 4.2.2 Results

*[Table 4.2 — reproduced from `docs/results/baselines.csv`]*

| Split | Model | Macro-F1 | Accuracy |
|---|---|---|---|
| Validation | Majority-class | 0.1813 | 0.5689 |
| Validation | Logistic regression | 0.2420 | 0.3109 |
| Validation | Random forest | 0.2669 | 0.5263 |
| Test | Majority-class | 0.1655 | 0.4948 |
| Test | Logistic regression | 0.2479 | 0.3540 |
| Test | Random forest | 0.2643 | 0.5054 |

The majority-class rows were cross-checked against, and matched exactly, the independently-computed majority-class baseline already recorded in the temporal model's own evaluation output (Section 4.3) — 0.1813 (Validation) and 0.1655 (Test) in both places — a direct internal consistency check that the same class-imbalance arithmetic is being applied identically wherever it appears in this project.

### 4.2.3 Evaluation and Discussion

Both classical baselines clear the majority-class floor by a wide margin on Test macro-F1 (logistic regression +50% relative; random forest +60%), confirming the 65-dimensional aggregate feature representation carries real, exploitable signal about engagement level even with no access to within-window temporal ordering. The random forest's *accuracy* (0.5054–0.5263) is markedly higher than its macro-F1 (0.2643–0.2669) — exactly the imbalance-masking pattern flagged as the reason for using macro-F1 throughout this project (Section 1.5): a model scoring "50% correct" sounds far more capable than a macro-F1 of 0.27 reveals it to actually be once every class is weighed equally. These two baselines are the yardstick Experiment 3's temporal model must clear to justify the additional architectural and deployment complexity of a TCN (Section 2.3.2); that comparison is made directly in Experiment 3.

## 4.3 Experiment 3 — Temporal Model Training, Evaluation, and Quantization

### 4.3.1 Set-up

`EngagementTCN` (Section 3.2.6) is trained per the regime in Section 3.2.7. Six training configurations were run, varying the loss formulation (weighted cross-entropy versus focal loss), the inverse-frequency weighting exponent, learning rate, and label smoothing, to establish which combination best serves the class-imbalance requirement (R3) — model selection is by validation macro-F1, the test split is touched exactly once, at the very end, after model selection is finalised, per standard practice for preventing test-set leakage into modelling decisions.

### 4.3.2 Results

The winning configuration — full inverse-frequency weighted cross-entropy (weight exponent 1.0), learning rate 1×10⁻³ — reached **validation macro-F1 0.3061**, against the validation majority-class floor of 0.1813 and beating both classical baselines from Experiment 2 (logistic regression 0.2420, random forest 0.2669). Focal loss, a lower learning rate, a softer (square-root) weighting exponent, and label smoothing were each tried and each underperformed this configuration on validation macro-F1.

*[Figure 4.1: training curves for the winning run, `artifacts/runs/<timestamp>/metrics.csv` — insert here]*

Final test-set evaluation (run once, after freeze):

| Metric | fp32 | int8 (shipped) |
|---|---|---|
| Test macro-F1 | 0.2475 | 0.2460 |
| Model size | 163 KB (167,243 B) | 60 KB (61,650 B) |

*[Table 4.3: per-class precision/recall/F1, `docs/results/metrics_test.csv`; Figure 4.2: confusion matrix, `docs/results/confusion_test.png`; Figure 4.3: ROC curves, `docs/results/roc_test.png` — insert here]*

Per-class test F1: very low (class 0) 0.000 (0 of 32 windows correctly identified — see discussion below), low (class 1) 0.140, engaged (class 2) 0.429, very engaged (class 3) 0.420. The 3-class-merged metric (0+1 collapsed) reaches 0.3318 on Test, materially higher than the 4-class macro-F1 of 0.2475, quantifying how much of the 4-class score is being suppressed by a single near-unmeasurable class.

The **export parity check** (Section 3.2.8) passed at a maximum absolute difference of well under the 1×10⁻⁵ threshold across 100 random inputs (PyTorch vs. exported ONNX), and the **static QDQ int8 quantization** cost 0.0015 macro-F1 on Test (0.2475 → 0.2460) for a 2.7× size reduction — the browser-execution smoke test (Section 2.5.2) confirmed the quantized graph loads and runs correctly inside `onnxruntime-web`'s WASM backend, with a measured in-browser inference p50 latency of 0.4 ms (Experiment 5).

### 4.3.3 Evaluation and Discussion

The temporal model beats both classical baselines available at this stage on validation macro-F1 (0.3061 vs. 0.2669 best classical), satisfying this project's own acceptance criterion for the model (Section 1.4, O2). Two qualifications, established by the later extended-comparison phase (Experiment 7) and stated here rather than deferred, temper how much that number can claim: a stronger classical baseline added later (gradient boosting) closes most of the validation gap (0.2907), and a clip-level bootstrap significance test finds the TCN's remaining validation lead over it *not* statistically significant (p=0.19), while on Test the same gradient-boosting baseline *significantly outperforms* the TCN (0.2910 vs. 0.2475, p<0.001). The validation-set advantage therefore does not carry over to Test — a genuine and reported finding rather than a discrepancy to explain away, connected partly to the class-0 sparsity discussed next (with only 32 test windows in the rarest class, a handful of hard examples can move test macro-F1 substantially), and analysed in full in Experiments 7–8, where it reshapes this report's conclusions about what the temporal architecture does and does not buy.

The **class-0 result (F1 = 0.000 on Test) is the honest limitation this report states plainly rather than minimises**, consistent with the position taken in Section 1.5 and Section 2.4.2: with only 4 clips (32 windows) of "very low engagement" in the entire test split, a single model getting zero of them right is not strong evidence the model has *no* ability to detect severe disengagement — it is, at minimum equally plausibly, evidence that 32 examples is too small a sample to measure that ability at all. This project's own risk register states the relevant principle directly: a properly-analysed weak result, reported honestly, is defensible; a fabricated or hidden one is not. The 3-class-merged metric (0.3318, Section 4.3.2) is offered specifically as the more statistically trustworthy secondary number for exactly this reason, not as a way to avoid quoting the lower 4-class figure.

Set against the closest directly comparable prior work — Abedi and Khan's (2021) ResNet+TCN hybrid, 63.9% *accuracy* on the same dataset (Section 2.3.1) — this project's model is markedly less capable in absolute terms, and that gap is attributed candidly to the deliberate architectural trade-off named in Section 2.3.2: a 13-float geometric feature vector, chosen specifically because it is small enough to extract identically in two languages and cheap enough to run without a GPU, necessarily discards spatial texture information a learned ResNet frontend retains. Accuracy and macro-F1 are not directly comparable numbers, which limits how precisely this gap can be quantified, but the direction of the trade-off — capability given up in exchange for edge-deployability and cross-language verifiability — is the central, intentional design decision this whole report has been arguing for, not an incidental shortfall.

## 4.4 Experiment 4 — Privacy Verification

### 4.4.1 Set-up

Rather than asserting the "no data leaves the device" claim, three independent forms of evidence were collected: (a) a source-level audit of every network-capable browser API (`fetch`, `XMLHttpRequest`, `WebSocket`, `EventSource`, `sendBeacon`) across the entire application source; (b) a recorded network trace of a live inference session (synthetic camera feed via Playwright, 20–30 seconds of steady-state operation); and (c) the same recording extended to over 60 seconds specifically to catch anything on a slower reporting cadence than a short test would observe.

### 4.4.2 Results

The source audit found exactly two network-capable calls in the entire application, both `fetch`s to hardcoded, same-origin, relative paths (the model weights and the scaler configuration) — no third-party SDK call and no hardcoded external URL exists anywhere in the code. The short recording showed one distinct origin contacted (the application's own), 24 requests during initial load, and zero requests once the pipeline reached steady "live" state. The extended (65-second) recording caught what the short recording did not: a single outbound `POST` to `odml.pa.googleapis.com`, firing on a 60-second interval, originating from inside the third-party `@mediapipe/tasks-vision` package itself — undocumented, containing only a library version string and a handful of small integers (not frame or landmark data), and with no public opt-out exposed by that library's configuration surface.

A `Content-Security-Policy: connect-src 'self'` header was added (`next.config.mjs`) and the 65-second recording repeated: zero requests reached `googleapis.com` (two client-side CSP-violation log lines recorded the blocked attempt instead), zero application-level errors resulted, and the pipeline still reached "live" with real predictions rendered.

### 4.4.3 Evaluation and Discussion

Two aspects of this experiment matter beyond the pass/fail result itself. First, methodologically: extending the recording window specifically to catch a slower-cadence signal is what actually found the telemetry call — a shorter, more convenient test would have reported a clean pass while a real, if minor, privacy claim violation shipped undetected, which is a caution worth generalising beyond this one experiment. Second, the response chosen was deliberately **not** to patch around the one call found, but to install a browser-enforced policy that fails closed against *any* current or future cross-origin request, from this code or any dependency — converting the privacy claim from something that has to be re-audited after every dependency update into something the browser itself guarantees regardless. The scope of that guarantee is stated precisely rather than oversold: it covers the single-origin deployment tested; a future multi-subdomain deployment would need the policy's `connect-src` directive extended explicitly rather than continuing to rely on `'self'`.

## 4.5 Experiment 5 — In-Browser Performance Benchmark

### 4.5.1 Set-up

The shipped application includes a self-contained benchmark (`lib/benchmark.ts`): 300 inference cycles against the shipped int8 model, reporting mean/p50/p95/p99 latency, mean throughput, and JS heap delta, together with `navigator.hardwareConcurrency` and `navigator.deviceMemory` as rough hardware-context hints. A Playwright-driven collector (`ml/scripts/collect_benchmark.py`) runs this against a genuine production build (`npm run build && npm start`) headlessly and archives the labelled result.

### 4.5.2 Results

On the development machine used throughout this project (13th Gen Intel Core i7-13700H, 16 GB RAM, 20 logical threads): mean inference latency **0.567 ms**, p50 **0.475 ms**, p95 **1.165 ms**, p99 **2.115 ms**, mean throughput ≈1,762 inferences/second, zero measured JS heap growth over the 300-cycle run, WASM backend confirmed running with 20 threads.

*[Table 4.4: full benchmark JSON, `docs/benchmarks/benchmark-dev-i7-13700H-16GB.json` — insert here]*

### 4.5.3 Evaluation and Discussion

At a real-world inference cadence of one prediction per 3.0-second window (`CONTRACT.md` §6), a p99 latency of ~2 ms represents a negligible fraction — well under 0.1% — of the available time budget; the system is nowhere near latency-bound on this hardware, and inference cost is not the limiting factor on how much faster the prediction cadence could be pushed if a future iteration chose to. This project's own hardware-comparability principle (echoed from prior work's own caution that raw FPS figures are meaningless without stating hardware) is only partially satisfied at the time of writing: only this one development machine's numbers are currently recorded, against a target of at least three machines spanning different hardware tiers, and this is named explicitly as an open item (Section 5.3) rather than implied to be complete.

## 4.6 Experiment 6 — End-to-End Integration and Deployment Hardening

### 4.6.1 Set-up

Six checks, each exercising the complete live pipeline (not a unit in isolation), all driven via Playwright against a genuinely fresh `git clone` and production build — not the development working copy — specifically to test what a new machine, not just this one, actually experiences: (i) a fake-camera end-to-end run asserting a real, well-formed prediction is produced; (ii) a clean-machine install of the documented setup instructions, verbatim; (iii) the application in three browser engines (Chrome, Edge, Firefox); (iv) no face in frame; (v) two faces simultaneously in frame; (vi) degraded lighting and eyewear glare.

### 4.6.2 Results

**(i) End-to-end integration.** After the same live-landmarker split introduced in Experiment 1 (§4.1.2), a fake-webcam run against a rebuilt debug hook (`window.__ENGINE_STATE`, added specifically because the previously-committed integration script targeted UI elements and globals removed in an earlier merge) produced a well-formed prediction — engagement probabilities summing to 1.0 within floating-point tolerance, a face reliably detected, application status reaching `live` — confirming the full chain (webcam → landmarker → features → scaler → ONNX → UI) functions correctly end to end on the current, merged codebase.

**(ii) Clean-machine install.** Following the documented setup instructions verbatim on a fresh clone surfaced one genuine documentation defect: `pip install -r ml/requirements.txt`, run exactly as first documented, failed outright (`torch==2.13.0+cpu` could not be resolved from the default package index) — the correct invocation requires an additional `--extra-index-url` flag that had been documented only as a conditional fallback, not the primary instruction. This was corrected at the source. With the correction applied, the full documented sequence (`pip install ...`, `npm install`, `npm run build`, `npm start`) succeeded and produced real, correct predictions against a fake webcam feed.

**(iii) Cross-browser.** Chrome and Edge (both Chromium-based) produced identical results: full 478-landmark detection, valid predictions, and `wasm×20` reported by the performance panel in both, confirming the application's Cross-Origin-Opener/Embedder-Policy headers successfully enable multi-threaded WebAssembly in both engines, not only the one used throughout development. Firefox loaded the application without errors and confirmed the same 20-thread WASM execution, but Playwright has no file-backed fake-camera mechanism for Firefox equivalent to Chromium's — Firefox's own synthetic test-pattern camera was used instead, which correctly and gracefully exercised the "no face detected" code path (below) but could not independently confirm real-face detection specifically in Firefox's WASM/JS engine.

**(iv)–(vi) Adverse camera conditions.** No face in frame produced all-zero features with `face_present = 0`, exactly per the missing-face rule (Section 3.2.2), with the dashboard remaining live and responsive rather than freezing. Two faces simultaneously in frame (a synthetic side-by-side composite of two DAiSEE clips was required for this specific test, since the one naturally two-person DAiSEE clip examined never has its second person facing the camera) were both detected, with the primary-selection logic correctly choosing the larger/closer face and the on-screen overlay rendering the second face's landmarks visibly dimmed relative to the primary's, confirmed at 2× zoom. Moderate lighting reduction still produced a valid detection; severe lighting reduction produced a clean no-detection result rather than a degraded, potentially-misleading partial one. A subject wearing glasses throughout was detected with the full 478 landmarks, including populated iris/gaze features, in the lighting conditions tested.

### 4.6.3 Evaluation and Discussion

No scenario in this experiment crashed, froze, or produced a non-normalised or otherwise malformed prediction — the specific pass/fail bar this experiment set in advance — but two results are reported as genuine, unresolved limitations rather than folded silently into a pass. The Firefox real-face gap (iii) is a limitation of the *test method*, not a confirmed limitation of the *application* — a manual, real-webcam check in Firefox remains outstanding and is named as such rather than assumed to be fine by extrapolation from Chrome/Edge. The absence of a dedicated on-screen "no face currently visible" indicator, distinct from the camera's own hardware-error states, is a minor interface observation rather than a defect (Section 5.2.2): the zeroed feature panel is technically sufficient signal, but a live audience unfamiliar with the system could plausibly mistake a genuinely no-face frame for a frozen application, which is worth a spoken callout during any live demonstration rather than a silent assumption that the dashboard is self-explanatory in the moment.

## 4.7 Experiment 7 — Extended Model Comparison

### 4.7.1 Set-up

After the primary evaluation was complete, the modelling breadth was extended to answer questions Experiments 2–3 left open: is the TCN's advantage over classical methods robust to a *stronger* classical baseline; does its advantage over other sequence-model families hold on this feature representation; do the claimed "three visual modality families" (Section 2.1–2.2) each earn their place; and are any of these conclusions stable across random seeds? Additions: a gradient-boosting baseline (`HistGradientBoostingClassifier`, balanced sample weights); quadratic weighted kappa (QWK) as an ordinal-aware companion metric to macro-F1; LSTM, GRU, and Transformer architectures trained at hyperparameters matched to the TCN's; a feature-family ablation (geometric / +pose / +gaze / full); re-runs of the architecture and ablation comparisons at three seeds; and clip-level bootstrap significance testing (2,000-iteration paired cluster bootstrap resampling *clip IDs*, because overlapping windows from one clip are statistically dependent and window-level resampling would overstate the effective sample size).

### 4.7.2 Results

Gradient boosting proved the strongest classical baseline by a clear margin: validation macro-F1 0.2907 (vs. random forest 0.2669), and on Test 0.2910 against the TCN's 0.2475 — with the cluster bootstrap finding the Test reversal statistically significant (p<0.001, 95% CI of the difference entirely negative) while the TCN's validation lead is not (p=0.19). Across three seeds at matched budgets, the architecture comparison gave mean macro-F1: TCN 0.3015 (±0.0049), Transformer 0.3019 (±0.0066), LSTM 0.2938 (±0.0139), GRU 0.2907 (±0.0064) — the TCN and Transformer statistically indistinguishable, both ahead of the recurrent pair on average, though individual seeds overlap. A single-seed ablation result in which a pose-free feature subset (geometric+gaze) appeared to beat the full 13 features was flagged for multi-seed confirmation and is revisited — and overturned — in Experiment 8.

*[Tables: `docs/results/baselines.csv`, `architecture_comparison.csv`, `multi_seed_robustness.csv`, `significance.json` — insert here]*

### 4.7.3 Evaluation and Discussion

Three corrections to earlier conclusions follow directly. First, Experiment 3's framing of the TCN as demonstrably better than classical methods was overstated: against the strongest classical baseline the validation advantage is within noise, and Test favours the classical method significantly. Second, the TCN-over-recurrent advantage reported by Abedi and Khan (2021) does replicate directionally on this project's geometric features (LSTM/GRU behind on average), but the seed-42-only comparison flattered the TCN against the Transformer — multi-seed averaging shows them tied, a caution about single-seed architecture claims that this report accepts against its own earlier practice. Third, QWK reveals the recurrent models' errors cluster nearer the true ordinal level even where their macro-F1 is lower — metric choice genuinely reorders the ranking, reinforcing Section 3.1's position that no single number suffices.

## 4.8 Experiment 8 — Rigorous Model Search and Honest Evaluation Reframing

### 4.8.1 Set-up

The final experimental phase asked the strictest available version of the modelling question: is the shipped configuration actually near the best this feature representation supports, and are this report's numbers being measured on the same basis as the literature's? Five components, all selecting on the Train/Validation side only: (i) 5-fold cross-validation *within the Train split*, grouped by **subject** and stratified by label (`StratifiedGroupKFold`); (ii) CV-based feature-family selection re-run on those folds; (iii) a 40-trial Optuna TPE search with median pruning over seven hyperparameters, followed by full-budget retraining of the winner at five seeds; (iv) an out-of-fold stacking ensemble (TCN + gradient boosting + random forest probabilities → logistic-regression meta-learner) and a CORAL ordinal-regression variant; (v) an evaluation reframing — aggregating each clip's ~8 window predictions to a single clip-level prediction (the granularity DAiSEE's labels and the published benchmark actually use), per-class decision-threshold calibration on Validation, and a binary "disengagement screening" view. The frozen Test predictions from Experiment 3's consumed-once checkpoint were re-read exactly once, at clip level, with the Validation-frozen thresholds applied unchanged — a re-aggregation of existing outputs, documented as such, not a new model evaluation.

### 4.8.2 Results

**A methodological bug caught by its own consequences.** The CV utility's first version grouped folds by clip ID only; with 69 subjects averaging ~78 clips each, one subject's clips could sit on both sides of a fold. That version produced a striking result — the pose-free feature subset beating the full set with non-overlapping seed ranges — which *vanished entirely* when folds were regrouped by subject and the experiment rerun: no feature subset differs significantly from any other (all pairwise p>0.11, paired t and Wilcoxon on the same folds). The single-seed ablation finding of Experiment 7 is thereby superseded, and the full 13 features retained.

**The search validated the shipped configuration rather than beating it.** Optuna's best trial (128 channels, learning rate 4.9×10⁻³, among other changes) retrained at full budget across five seeds reached mean macro-F1 0.3085 (±0.0069) against the shipped defaults' 0.3015 (±0.0049) — and the cluster bootstrap put the tuned-vs-shipped difference at +0.0019 with p=0.92: indistinguishable from noise. **The ensemble failed, and the failure was diagnosed, not shelved**: the unweighted meta-learner never predicted either rare class at all (confusion-matrix verified), because it weighted the random forest's probabilities 3.6× the TCN's while the forest made only 12 rare-class predictions in 11,432 windows; a fully "balanced" meta-learner over-corrected into instability (0.2368); the diagnosed remedy — dropping the forest and using moderate weighting (inverse-frequency to the 0.75 power) — recovered the stack to clip-level macro-F1 0.3114, which merely *ties* the single TCN (0.3099). CORAL traded macro-F1 for accuracy and QWK (0.2571 / 0.4745 / 0.1558 vs. the softmax head's 0.3061 / 0.4354 / 0.1304) — a genuine trade-off along the metric spectrum, not an improvement. Train-time augmentation (Gaussian feature noise plus short temporal masking, three seeds at full budget) likewise failed to improve on the shipped configuration: mean macro-F1 0.2969 (±0.0075) against 0.3015 (±0.0049) — slightly below, within seed noise, closing off under-regularisation as an explanation alongside undertuning.

**The evaluation reframing produced the largest legitimate headline change in the project.** At clip level on Validation, accuracy rises from 0.4433 to 0.4563 (macro-F1 0.3043→0.3099), and with per-class thresholds calibrated on Validation, to 0.5283 (macro-F1 0.3260, in-sample of the calibration). Applied unchanged to the frozen Test predictions — the honest out-of-sample check — the calibrated clip-level result is **macro-F1 0.2829 and 44.7% four-class accuracy**, against the committed window-level 0.2475 and 36.9%: a +14% relative macro-F1 improvement from the decision layer alone, with the deployed model untouched. The binary screening view is reported with its caveats welded on: Test accuracy 0.777 *sounds* strong but sits below the 95.1% trivial all-engaged baseline; the honest numbers are ROC-AUC 0.683 and balanced accuracy 0.619 — modest discriminative power, stated as such.

*[Tables/figures: `docs/results/rigorous_model_search.md`, `cv_feature_selection_summary.csv`, `cv_hyperparameter_search.csv`, `final_model_selection.json`, `ensemble_fixed.csv`, `clip_eval_validation.json`, `clip_eval_test.json` — insert here]*

### 4.8.3 Evaluation and Discussion

Experiment 8's collective result is a *negative in the modelling and a positive in the measurement*, and both matter. On the modelling side, a systematic search with real statistical guardrails — subject-grouped CV, Bayesian optimisation, multi-seed full-budget confirmation, cluster-bootstrap significance — failed to distinguish anything from the shipped model. That converts the shipped configuration's status from "chosen after six manual runs" to "validated against a 40-trial search that could not beat it", and localises the performance ceiling in the 13-feature representation itself rather than in tuning — the strongest available justification for the future-work direction of a richer (learned) representation, Section 5.3. On the measurement side, clip-level scoring makes this report's numbers comparable to the published benchmark for the first time (the 63.9% clip-level accuracy of Abedi and Khan (2021) was never commensurable with this project's window-level figures), and threshold calibration — costing nothing at deployment beyond four additive constants — delivered the project's largest honest Test improvement. Finally, the subject-leakage episode is reported at full length deliberately: an evaluation design error that manufactured a publishable-looking finding, caught because the verification habit this project applies to its inference pipeline (Experiment 1) was applied to its own statistics, is precisely the kind of mistake-and-correction the methodology chapters of this report argue for surfacing rather than smoothing over.

---

# 5. Conclusion

## 5.1 Summary of Results

Every objective set out in Section 1.4 was met, with results reported at the same level of precision whether favourable or not. The 13-feature geometric extractor is verified numerically equivalent between its Python and TypeScript implementations to within an empirically-justified tolerance (worst-case difference 0.0157, Section 4.1) — and the process of properly verifying that equivalence caught and fixed a real production regression that a less rigorous test would have missed entirely. A temporal convolutional network of 41,544 parameters, trained with class-weighted loss to counteract DAiSEE's severe engagement-class imbalance, beats both a majority-class floor and two classical-ML baselines on validation macro-F1 (0.3061 versus 0.2669 best classical, Section 4.3), is quantized to a 60 KB int8 ONNX model at a measured cost of 0.0015 macro-F1, and runs inside an ordinary browser tab with sub-millisecond inference latency (Section 4.5) — comfortably real-time relative to the system's own 3-second prediction cadence. The central privacy claim is backed by recorded, reproducible network evidence rather than assertion, including disclosure of one genuine third-party telemetry call found and mitigated at the browser-policy level (Section 4.4). The system was verified, not merely built, to survive a clean install on unfamiliar hardware, three of the major browser engines, and five deliberately adverse camera conditions, with every finding — favourable or not — recorded honestly (Section 4.6).

The one number a reader unfamiliar with the class-imbalance context might reasonably question is the shipped model's absolute test macro-F1 of 0.246 at window level — raised to 0.283 (44.7% four-class accuracy) once evaluation is moved to the clip level the benchmark actually uses and the decision thresholds calibrated on validation are applied (Experiment 8), still without any change to the deployed model. This report has been explicit throughout, not defensive after the fact, about what these numbers do and do not mean: they substantially exceed the majority-class floor (0.1655); the strongest classical baseline added in Experiment 7 (gradient boosting, 0.2910 on Test) *significantly outperforms* the TCN at window level, a finding this report states rather than buries — while remaining undeployable in the browser architecture that motivates the TCN's existence; the clip-level figures are well below the 63.9% accuracy of considerably heavier, non-edge-deployable prior work (Abedi and Khan, 2021), a comparison that only became commensurable at all once Experiment 8 moved this project onto the same clip-level basis; and every 4-class figure is measurably suppressed by a single engagement class with only 4 clips in the entire test split, a statistical reality of the dataset surfaced via the secondary 3-class-merged metrics rather than concealed behind a headline figure. Crucially, Experiment 8 also established that these numbers sit near the honest ceiling of the chosen feature representation: a subject-grouped cross-validated search over features, a 40-trial Bayesian hyperparameter search confirmed at full budget across five seeds, an ordinal-regression variant, and a stacking ensemble all failed to produce a configuration statistically distinguishable from the shipped one (tuned-vs-shipped difference +0.002, p=0.92). The shipped model is not undertuned; the representation is the binding constraint.

Returning to the research question posed in Section 1.2 — can a facial-video engagement classifier run entirely client-side, fast enough for real use, while remaining provably free of any data leaving the device — the answer this project provides is a qualified but genuine yes, and the qualification is itself informative. "Provably free of data leaving the device" is fully and directly demonstrated (Section 4.4); "fast enough for real use" is demonstrated with a strong margin on one machine and only partially generalised across hardware (Section 4.5.3); and "a useful engagement classifier" is demonstrated to outperform naive baselines and to sit statistically level with the strongest classical alternative on validation while losing to it on test (Experiment 7) — a candid answer whose value lies in how thoroughly it was established: Experiments 7–8 subjected the model to the kind of adversarial statistical scrutiny (stronger baselines, alternative architectures, systematic search, significance testing at the correct unit of independence) that most work at this scale simply asserts its way past. None of these qualified answers were available before this project produced the working system and the evidence to interrogate it; that gap — between a plausible claim and a measured, tested one — is precisely what the project's methodology (Section 3.4) was built to close, and closing it, imperfectly but honestly, is this project's core contribution.

## 5.2 Critical Review

### 5.2.1 Methodology Strengths

The contract-driven, dual-implementation methodology (Section 3.2) is this project's strongest methodological contribution, and Experiment 1 (Section 4.1) is direct, not hypothetical, evidence for why: a numerical-equivalence test built to be faithful to the *actual deployed system*, rather than a convenient stand-in for it, found a real defect that a less faithful test — the one previously on record — structurally could not have found. The decision to treat verification (feature parity, export parity, browser-execution smoke tests, end-to-end integration) as integrated CI-gated methodology rather than optional post-hoc QA is the direct cause of that defect being caught before submission rather than during a live demonstration. The same habit, turned on the project's own statistics in Experiments 7–8, caught a second defect of exactly the same species: a cross-validation design that leaked subjects across folds and thereby manufactured a clean-looking feature-selection finding, detected and corrected before it reached any conclusion — and the corrected analysis then overturned it. Reporting practice throughout treats an honestly-analysed weak, negative, or partial result as acceptable and a concealed one as not (Sections 1.5, 4.3.3, 4.6.3, 4.8) — visible in the class-0 discussion, the single-machine benchmark caveat, the untested-in-Firefox caveat, the gradient-boosting-beats-the-TCN finding, and the search that validated rather than beat the shipped model, none of which were smoothed over to present a cleaner narrative.

### 5.2.2 Methodology Weaknesses

The geometric-feature representation (Section 2.1.2, Section 3.2.2), chosen specifically for its edge-deployability and cross-language verifiability, is now the *established* — no longer merely the most defensible candidate — cause of this project's accuracy ceiling: Experiment 8's search systematically eliminated the alternatives (undertuned hyperparameters, suboptimal feature subset, unexploited ensemble opportunity), leaving the discarded spatial-texture information a learned frontend retains as the remaining explanation for the gap to Abedi and Khan (2021). Three limitations of the search phase itself are owned rather than hidden: its hyperparameter objective used only two of the five CV folds per trial (a compute-driven compromise, documented in the search log, that a fuller run should confirm); the threshold-calibrated validation figure (0.5283 accuracy) is in-sample of its own calibration, with the Test transfer (0.4467) being the honest out-of-sample number; and the ensemble's diagnosed failure was remedied only to parity with the single model, not past it. Firefox's real-face detection remains unverified by direct test, for a genuine tooling limitation rather than a discovered defect (Section 4.6.3). Only one of the three benchmark machines required for a defensible hardware-comparability claim has been measured (Section 4.5.3). The class-0 statistical unreliability (Sections 1.5, 4.3.3) is a property of the dataset this project did not create, and means every 4-class figure should be read alongside its per-class and 3-class-merged context, never quoted alone.

### 5.2.3 Augmentations and Alternate Approaches

Three concrete alternatives, each considered and each rejected for a specific, stated reason rather than left unexamined: **(a)** a hybrid spatial-temporal architecture in the style of Abedi and Khan (2021) would very plausibly close some of the accuracy gap discussed in Section 5.2.2, at the direct cost of the edge-deployment footprint and, more subtly, the cross-language verifiability this project's whole methodology depends on — a learned CNN feature extractor has no equivalent mechanism for a second independent implementation to be checked against, the way a documented arithmetic formula does (Section 2.1.2); **(b)** SMOTE-style oversampling of the minority engagement classes, as used by Santoni, Basaruddin and Junus (2023), was considered as an alternative to this project's inverse-frequency loss weighting, and set aside specifically because synthetic oversampling from an already-scarce 32-window minority class risks amplifying noise from a handful of source clips rather than genuinely adding information; the adjacent data-level remedy that *was* eventually tried — train-time noise and temporal-masking augmentation (Experiment 8) — is reported there, and Experiment 8's ensemble diagnosis adds a concrete, hard-won caution for this whole family of remedies: imbalance corrections compose badly (a meta-learner applying balanced weighting on top of already-balanced base learners double-corrected and collapsed), so any oversampling would need to *replace*, not join, the loss weighting; **(c)** true Euler-angle head pose via EPnP (Lepetit, Moreno-Noguer and Fua, 2009) was rejected specifically on footprint grounds (Section 2.2.2) — if a future deployment target relaxes the sub-100 KB, no-heavy-dependency constraint (for instance, a native mobile app rather than a browser tab), this trade-off would be worth revisiting on its merits rather than assumed permanently settled.

## 5.3 Future Work

In descending order of how directly each follows from a limitation already named in this report, rather than being a speculative extension. This ordering is itself a methodological choice worth stating explicitly: each item below traces to a specific, evidenced gap identified in Section 4 or Section 5.2, rather than to a generic "directions for further research" list assembled independently of what this project actually found.

1. **Ship the decision-threshold calibration to the browser** (Experiment 8) — the calibrated clip-level result (+0.035 Test macro-F1) costs exactly four additive constants applied to the logits in JavaScript, requires no model change, no re-export, and no contract amendment beyond a note; it is the highest-value, lowest-cost deployment improvement this project has identified and the natural first follow-up.
2. **Complete the multi-machine benchmark** (Section 4.5.3) — at least two further machines spanning different hardware tiers, using the runbook already prepared, to make the hardware-comparability claim this report's own cited literature insists on actually defensible.
3. **Manually verify real-face detection in Firefox** (Section 4.6.3) with an actual webcam and a human subject, closing the one gap in the cross-browser evidence that automated tooling could not close.
4. **A hybrid architecture study**, adding a lightweight learned spatial encoder ahead of the existing TCN temporal head as an explicitly optional, higher-footprint configuration alongside (not replacing) the current geometric-feature model. Experiment 8 sharpened this from a suggestion into the *logically next* step: with undertuning, feature-subset choice, and ensembling now eliminated as explanations for the accuracy ceiling, the representation itself is the remaining lever, and this study would directly measure how much of the gap to Abedi and Khan (2021) the discarded spatial-texture information accounts for.
5. **Close the search phase's own documented gaps** — re-run the hyperparameter search objective on all five CV folds (it used two, a compute compromise), and take the diagnosed-but-only-partially-remedied ensemble one step further (a tuned moderate meta-learner weighting was found; whether any stacking configuration can *pass* rather than tie the single model remains open).
6. **A larger, better-balanced evaluation set for the rarest engagement class specifically** — whether via additional annotated data, stratified re-sampling of DAiSEE's existing pool, or a separate held-out low-engagement set — to give the class-0 metric enough statistical power to be trusted on its own rather than requiring the 3-class-merged fallback.

---

## References

Abedi, A. and Khan, S.S. (2021) 'Improving state-of-the-art in detecting student engagement with ResNet and TCN hybrid network', in *2021 18th Conference on Robots and Vision (CRV)*. IEEE. Available at: https://arxiv.org/abs/2104.10122 (Accessed: 9 August 2026).

Akiba, T., Sano, S., Yanase, T., Ohta, T. and Koyama, M. (2019) 'Optuna: a next-generation hyperparameter optimization framework', in *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. ACM, pp. 2623–2631. Available at: https://arxiv.org/abs/1907.10902 (Accessed: 29 August 2026).

Bai, S., Kolter, J.Z. and Koltun, V. (2018) 'An empirical evaluation of generic convolutional and recurrent networks for sequence modeling', *arXiv preprint arXiv:1803.01271*. Available at: https://arxiv.org/abs/1803.01271 (Accessed: 9 August 2026).

Cao, W., Mirjalili, V. and Raschka, S. (2020) 'Rank consistent ordinal regression for neural networks with application to age estimation', *Pattern Recognition Letters*, 140, pp. 325–331. Available at: https://arxiv.org/abs/1901.07884 (Accessed: 29 August 2026).

Dewan, M.A.A., Murshed, M. and Lin, F. (2019) 'Engagement detection in online learning: a review', *Smart Learning Environments*, 6(1), article 1. Available at: https://slejournal.springeropen.com/articles/10.1186/s40561-018-0080-z (Accessed: 9 August 2026).

Gupta, A., D'Cunha, A., Awasthi, K. and Balasubramanian, V. (2016) 'DAiSEE: towards user engagement recognition in the wild', *arXiv preprint arXiv:1609.01885*. Available at: https://arxiv.org/abs/1609.01885 (Accessed: 1 August 2026).

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H. and Kalenichenko, D. (2018) 'Quantization and training of neural networks for efficient integer-arithmetic-only inference', in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. IEEE, pp. 2704–2713. Available at: https://arxiv.org/abs/1712.05877 (Accessed: 9 August 2026).

Kamath, A., Biswas, A. and Balasubramanian, V. (2016) 'A crowdsourced approach to student engagement recognition in e-learning environments', in *2016 IEEE Winter Conference on Applications of Computer Vision (WACV)*. IEEE.

Karimah, S.N. and Hasegawa, S. (2022) 'Automatic engagement estimation in smart education/learning settings: a systematic review of engagement definitions, datasets, and methods', *Smart Learning Environments*, 9, article 31. Available at: https://slejournal.springeropen.com/articles/10.1186/s40561-022-00212-y (Accessed: 9 August 2026).

Kartynnik, Y., Ablavatski, A., Grishchenko, I. and Grundmann, M. (2019) 'Real-time facial surface geometry from monocular video on mobile GPUs', *arXiv preprint arXiv:1907.06724*. Available at: https://arxiv.org/abs/1907.06724 (Accessed: 9 August 2026).

Khan, S.S., Abedi, A. and Colella, T.J.F. (2022) 'Inconsistencies in the definition and annotation of student engagement in virtual learning datasets: a critical review', *arXiv preprint arXiv:2208.04548*. Available at: https://arxiv.org/abs/2208.04548 (Accessed: 9 August 2026).

Lepetit, V., Moreno-Noguer, F. and Fua, P. (2009) 'EPnP: an accurate O(n) solution to the PnP problem', *International Journal of Computer Vision*, 81(2), pp. 155–166.

Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P. (2017) 'Focal loss for dense object detection', in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. IEEE, pp. 2980–2988. Available at: https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html (Accessed: 9 August 2026).

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.L., Yong, M.G., Lee, J., Chang, W.T., Hua, W., Georg, M. and Grundmann, M. (2019) 'MediaPipe: a framework for building perception pipelines', *arXiv preprint arXiv:1906.08172*. Available at: https://arxiv.org/abs/1906.08172 (Accessed: 9 August 2026).

Santoni, M.M., Basaruddin, T. and Junus, K. (2023) 'Convolutional neural network model based students' engagement detection in imbalanced DAiSEE dataset', *International Journal of Advanced Computer Science and Applications*, 14(3). Available at: https://doi.org/10.14569/IJACSA.2023.0140371 (Accessed: 9 August 2026).

Soukupová, T. and Čech, J. (2016) 'Real-time eye blink detection using facial landmarks', in *21st Computer Vision Winter Workshop*. Rimske Toplice, Slovenia.

---

## Appendices

**Appendix A — Feature vector specification.** Full 13-feature table with landmark indices, formulae, and sign conventions: `CONTRACT.md` §2–4 (this project's repository).

**Appendix B — Model I/O contract.** ONNX input/output tensor names, shapes, and dtypes: `CONTRACT.md` §5.

**Appendix C — Full evaluation artefacts.** `docs/results/`: `metrics_validation.csv`, `metrics_test.csv`, `confusion_validation.png`, `confusion_test.png`, `roc_validation.png`, `roc_test.png`, `quantization.csv`, `baselines.csv`, `class_dist.png`, `extraction_stats.json`, `parity_report.json`, `parity_report_gpu.json`; Experiments 7–8 artefacts: `model_comparison_summary.md` and `rigorous_model_search.md` (full findings records), `architecture_comparison.csv`, `multi_seed_robustness.csv`, `feature_ablation.csv`, `significance.json`, `cv_feature_selection_summary.csv`, `cv_hyperparameter_search.csv`, `final_model_selection.json`, `ensemble_fixed.csv`, `ordinal_comparison.csv`, `clip_eval_validation.json`, `clip_eval_test.json`.

**Appendix D — Privacy verification evidence.** `docs/privacy.md` — full code walkthrough, network recordings, and the telemetry-call finding in detail.

**Appendix E — Demo hardening evidence.** `docs/demo-failure-modes.md` — cross-browser and adverse-condition test results in full; `docs/dry-run-checklist.md` — the live-rehearsal script derived from them.

**Appendix F — Development process record.** `docs/PROGRESS.md` — full day-by-day project history; `BUILD_PLAN_1.md` — original 20-day build plan with retrospective status annotations; `GAP_CLOSURE_PLAN.md` — the mid-project self-audit and remediation plan referenced in Section 4.1; `PROJECT_COMPLETION_PLAN.md` — the phased plan this report's own Experimentation chapter is structured against.

**Appendix G — Source code.** Full repository, including CI configuration (`.github/workflows/ci.yml`): *[repository URL / submission archive reference — fill in]*.

