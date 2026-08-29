# Honest evaluation reframing — clip-level scoring, threshold calibration, binary screening

Written 2026-08-29. Companion to `rigorous_model_search.md` (the search that validated
the shipped model) and `model_comparison_summary.md` (the extended comparison). This
file records the evaluation-side work: no model was changed anywhere below — every
improvement comes from measuring the *same frozen predictions* on the correct basis.

**Framing constraint, stated first**: this pass was commissioned with a request for
"at least 80% accuracy". That is not honestly reachable: published SOTA on DAiSEE
4-class engagement is 63.9% accuracy (Abedi & Khan 2021, heavy ResNet+TCN, not
edge-deployable), and this project's search phase established the 13-feature
representation's ceiling directly. Everything below is the honest maximum instead —
and the one configuration where an "80%+" style number *could* be quoted (binary
screening accuracy) is shown to sit **below its own trivial baseline**, which is why
it is never quoted without AUC and balanced accuracy attached.

## 1. Clip-level evaluation (`ml/src/clip_eval.py`)

DAiSEE labels are per-clip; the published benchmark is per-clip; this project had only
ever scored per-window (~8 overlapping windows per clip). Clip-level scoring (mean of
window softmax probabilities → argmax) makes the numbers commensurable with the
literature for the first time. Cross-check built in: the script recomputes window-level
macro-F1 and asserts it matches the committed `metrics_{split}.csv` before reporting
anything (passed: 0.3043 Validation, 0.2475 Test).

| Basis | Validation acc | Validation macro-F1 | Test acc | Test macro-F1 |
|---|---|---|---|---|
| Window-level (committed) | 0.4433 | 0.3043 | 0.3686 | 0.2475 |
| Clip-level, mean-prob | 0.4563 | 0.3099 | 0.3638 | 0.2482 |
| Clip-level + threshold calibration | 0.5283¹ | 0.3260¹ | **0.4467** | **0.2829** |

¹ Validation figures are in-sample of the calibration (offsets tuned on Validation);
the Test column is the honest out-of-sample transfer.

Checkpoint bookkeeping: Validation uses the shipped checkpoint
(`20260815_181604`); Test re-aggregates the frozen consumed-once checkpoint
(`20260801_185630`) — a re-aggregation of existing predictions, not a new model
evaluation, same precedent as `significance.py`. Details: `clip_eval_validation.json`,
`clip_eval_test.json`.

## 2. Per-class decision-threshold calibration

Coordinate ascent over four per-class log-probability offsets, maximising clip-level
macro-F1 **on Validation only**; the resulting offsets ([−0.6, −0.7, 0.0, 0.0] —
boosting the two rare classes) were then frozen and applied unchanged to Test.
**Result: Test macro-F1 0.2475 → 0.2829 (+14% relative), accuracy 36.9% → 44.7%** —
the largest legitimate Test-side improvement in the project, from a decision layer
costing four additive constants. Caveat carried in the JSON: the offsets were tuned on
the shipped checkpoint but applied to the frozen Test checkpoint (same architecture
and hyperparameters, different states-head training run). Shipping this calibration
to the browser is future-work item 1 in the thesis — it requires no re-export and no
contract change beyond a note.

## 3. Binary disengagement screening

Classes {0,1} → disengaged (the positive, rare, decision-relevant class), {2,3} →
engaged; score = P(0)+P(1), clip level.

| Split | Prevalence (disengaged) | Accuracy | Trivial all-engaged baseline | Balanced acc | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Validation | 11.6% | 0.7845 | 0.884 | 0.5955 | 0.6396 | 0.349 | 0.842 |
| Test | 4.9% | 0.7769 | 0.951 | 0.6187 | 0.6834 | 0.443 | 0.794 |

**The headline lesson, stated plainly**: binary accuracy of ~78% *sounds* strong and
is **below the trivial baseline on both splits** — exactly the accuracy-under-imbalance
trap the thesis's Section 1.5 describes. The honest numbers are the AUCs (0.64 / 0.68:
modest, real, above-chance discriminative power) and balanced accuracy. This table
exists partly as a worked example of why this project refuses to headline accuracy.

## 4. Ensemble fix (`ml/src/ensemble_fix.py`)

Implements both remedies from the committed ensemble diagnosis
(`rigorous_model_search.md`): drop the random forest and/or use moderate meta-learner
class weighting ((1/freq)^p). Full 2×5 grid on the cached OOF matrix, selected on
Validation clip-level macro-F1 (`ensemble_fixed.csv`):

- The diagnosis's predicted behaviour reproduces exactly: unweighted collapses
  (0.2635 clip macro-F1), fully balanced destabilises (0.2291), moderate p=0.75 is
  best at every stack composition, and dropping RF helps at every weight level.
- Best fixed stack (TCN+GBM, p=0.75): clip-level macro-F1 **0.3114**, accuracy 0.5017
  — which **ties, does not beat**, the single shipped TCN at clip level (0.3099).
  Stacking is now rescued from "worse than everything" to "adds nothing", and the
  single-model deployment stands on the merits.

## 5. Augmentation retrain (`train.py --augment`)

Train-time-only Gaussian feature noise (σ=0.05) plus short temporal masking (30% of
samples, 1–3 frame span replaced by the window's time-mean; no oversampling — the
ensemble diagnosis showed imbalance corrections compose badly with the weighted loss).
Three seeds at full production budget, against the shipped configuration's multi-seed
reference (mean 0.3015, std 0.0049, window-level Validation):

| Seed | Macro-F1 | Accuracy | QWK |
|---|---|---|---|
| 42 | 0.2974 | 0.4469 | 0.1212 |
| 7 | 0.3040 | 0.4327 | 0.1560 |
| 123 | 0.2891 | 0.4076 | 0.1259 |
| **mean ± std** | **0.2969 ± 0.0075** | — | — |

**Verdict: no improvement** — the augmented mean (0.2969) sits slightly *below* the
shipped configuration's (0.3015), well within the seed noise of both. A clean negative
result, consistent with everything else in this pass: the model is not
regularisation-starved any more than it is undertuned; the feature representation is
the constraint. Run dirs: `artifacts/runs/20260829_*_aug*`.

## Bottom line

No honest configuration of this system reaches "80% accuracy" in a meaningful sense,
and this document is the receipt for having looked properly. What the reframing did
deliver, all with the deployed model untouched: Test macro-F1 0.2475 → 0.2829 and
accuracy 36.9% → 44.7% on the benchmark's own clip-level basis via a four-constant
calibrated decision layer; a fair, first-time-commensurable comparison against the
63.9% SOTA; a repaired-and-closed ensemble question; and a worked demonstration of
why accuracy is not this project's headline metric.
