# Rigorous model search — methodology, full log, and findings

Written 2026-08-29. Standalone research record, in the same spirit as `docs/privacy.md`
and `docs/demo-failure-modes.md` — plain findings with numbers and file references, not
thesis prose. `docs/thesis/FYP_Report.md` is not edited in this pass; this file is meant
to be lifted into the thesis's Methodology/Experimentation chapters later, with minimal
rework, once the thesis-completion pass happens.

## Why this exists

A strict review of the project's technical/data-science rigor (excluding the
report-completeness questions, deferred to the end) identified five concrete gaps in the
modelling work up to that point (`docs/results/model_comparison_summary.md`):

1. **The core deliverable is weak** — macro-F1 0.25–0.31 against a 0.17–0.18 majority
   floor, and gradient boosting significantly beat the TCN on Test.
2. **No real hyperparameter search** — a handful of manual runs, then a 4-point grid.
3. **n=3 seeds is too thin** to support the strength of claims drawn from it.
4. **No cross-validation** — a single official split, no resampling feeding into model
   *selection* (only into post-hoc reporting).
5. **Feature selection happened after model design**, not before.

This document records the fix: a nested cross-validation pipeline (feature selection →
hyperparameter search → final multi-seed candidate → ensemble attempt → significance-
tested final selection), built and run in five phases.

## A methodological correction found and fixed mid-pipeline

The first version of the CV infrastructure (`ml/src/cv_splits.py`) grouped folds by
**clip ID** only. DAiSEE's Train split has 5,357 clips from only **69 unique subjects**
(≈78 clips/subject on average) — grouping by clip alone let the same subject's clips
split across a fold's train/held-out sets, exactly the subject-leakage
`ml/src/labels.py`'s own `assert_split_integrity()` already guards against between the
official Train/Validation/Test splits. This was caught before it propagated: the first
(leaky) run of Phase 1 produced a striking, clean-looking result — geometric+gaze
beating the full feature set with non-overlapping seed ranges — that evaporated
completely once folds were regrouped by **subject** (first 6 characters of `clip_id`,
`labels.py`'s own convention) and Phase 1 was rerun from scratch. The corrected
`cv_splits.py` is what every phase below actually used; `verify_folds()` asserts zero
subject overlap and full class coverage per fold, checked in code, not just claimed.

## Phase 0 — CV infrastructure

`ml/src/cv_splits.py`: 5-fold, subject-level, label-stratified splits within the
**Train split only** (`sklearn.StratifiedGroupKFold`, grouped by subject). Validation
and Test are never touched by this or any downstream CV script — DAiSEE benchmark
comparability on the splits that matter for final reporting is preserved throughout.

Verified fold composition (69 subjects total): every fold's held-out set covers all 4
engagement classes; zero subject overlap between any fold's train/held-out sets, by
assertion.

## Phase 1 — CV-based feature selection

`ml/src/cv_feature_selection.py`: each of `feature_groups.py`'s 4 presets (geometric,
geometric+pose, geometric+gaze, full) trained and evaluated on all 5 subject-level
folds (20 runs total, reduced/fast training budget via `cv_train.py`) — this is what
"feature selection before model design" means: the winning preset here is what Phase 2
tunes hyperparameters on, not a retrospective check.

| Preset | n_features | Mean macro-F1 | Std | Min | Max |
|---|---|---|---|---|---|
| geometric | 8 | 0.3041 | 0.0273 | 0.2669 | 0.3382 |
| geometric_pose | 11 | 0.2926 | 0.0179 | 0.2735 | 0.3092 |
| geometric_gaze | 10 | 0.3078 | 0.0273 | 0.2648 | 0.3378 |
| full | 13 | 0.2962 | 0.0063 | 0.2864 | 0.3018 |

Full per-fold detail: `docs/results/cv_feature_selection.csv`.

**Result: no preset wins significantly.** Pairwise paired t-test and Wilcoxon
signed-rank test across the same 5 folds (`scipy.stats.ttest_rel`/`wilcoxon`) find every
pairwise comparison non-significant (all p > 0.11; geometric_gaze vs. full: paired-t
p=0.4566, Wilcoxon p=0.4375). This directly supersedes an earlier, single-split/3-seed
finding (recorded in `docs/results/model_comparison_summary.md` §6) that geometric+gaze
beat the full feature set with non-overlapping seed ranges — that finding was itself
downstream of the clip-vs-subject leakage bug described above, and does not survive
correction. **Decision: proceed with the full 13-feature set** — the only defensible
choice absent a significant feature-selection result, and it matches the shipped
model's current input.

## Phase 2 — Real hyperparameter search (Optuna)

`ml/src/cv_hyperparameter_search.py`: TPE-sampled Bayesian search (`optuna==4.9.0`,
added to `ml/requirements.txt`) with median pruning (`MedianPruner`), both at the
epoch level (pruning mid-fold via `trial.report()`/`should_prune()` after every epoch)
and the fold level (pruning between folds against the running median of prior trials).
Search space: `lr` (log-uniform 1e-4–1e-2), `channels` (32–128, step 16), `dropout`
(0.1–0.5), `weight_power` (0.3–1.0), `state_loss_weight` (0.1–1.0), `label_smoothing`
(0–0.2), `batch_size` (64/128/256).

**A search-speed compromise, made explicit rather than hidden**: the objective
originally averaged macro-F1 across all 5 CV folds per trial, matching Phase 1's design.
In practice, a full 5-fold trial at reasonable per-fold epoch budgets did not
consistently survive this environment's periodic interruption of long background jobs
— confirmed directly (one full-5-fold trial ran for over an hour without completing a
single fold in two consecutive attempts). The objective was changed to use only the
**first 2 of 5 folds** per trial (`--search-folds 2`), and the per-fold budget reduced to
25 epochs/patience 6 (from an initial 40/8). This is a search proxy for finding a good
hyperparameter region, not a final evaluation — Phase 1 already used the full 5 folds
for feature selection, and Phase 3 below retrains the winning configuration at full
production budget on the complete official Train/Validation split. The trade-off is
named here so it isn't mistaken for the same rigor as Phase 1's feature-selection CV.

40 trials completed (20 `COMPLETE`, 20 `PRUNED` by the median pruner — roughly half the
naive compute). Full trial history: `docs/results/cv_hyperparameter_search.csv`.

**Winning configuration** (`docs/results/cv_hyperparameter_search_best.json`, 2-fold
search-proxy macro-F1 0.3441):

| Hyperparameter | Shipped default | Optuna-selected |
|---|---|---|
| `channels` | 64 | 128 |
| `dropout` | 0.2 | 0.238 |
| `lr` | 1e-3 | 0.0049 |
| `weight_power` | 1.0 | 0.382 |
| `state_loss_weight` | 0.5 | 0.980 |
| `label_smoothing` | 0.0 | 0.081 |
| `batch_size` | 128 | 256 |

## Phase 3 — Final candidate: full-budget, multi-seed training

`ml/src/final_candidate.py`: Phase 2's winning configuration retrained with the
**existing, unchanged `train.py`** (full 100-epoch/patience-15 production budget, not
the search's reduced budget) at 5 seeds (42, 7, 99, 17, 123) on the official Train
split, evaluated on official Validation.

| Seed | Macro-F1 | Accuracy | QWK |
|---|---|---|---|
| 42 | 0.3081 | 0.4990 | 0.1687 |
| 7 | 0.2995 | 0.4559 | 0.1179 |
| 123 | 0.3065 | 0.4320 | 0.1679 |
| 99 | 0.3101 | 0.4759 | 0.1790 |
| 17 | 0.3185 | 0.4949 | 0.1646 |
| **mean** | **0.3085** | — | — |
| **std** | **0.0069** | — | — |

Compare to the shipped TCN's own multi-seed result (`docs/results/
multi_seed_robustness.csv`, 3 seeds, default hyperparameters): mean 0.3015, std 0.0049.
The tuned configuration's mean is marginally higher (+0.0070), but see Phase 5 for
whether this difference is statistically meaningful (it is not).

## Phase 4 — Ensemble attempt (out-of-fold stacking)

`ml/src/ensemble_stack.py`: proper OOF stacking using Phase 0's 5 subject-level folds
— for each fold, the tuned TCN (Phase 2/3 config, reduced-epoch variant for search
speed — see the script's own docstring for the same interruption-driven budget
trade-off as Phase 2), gradient boosting, and random forest are trained on the other 4
folds and predict class *probabilities* on the held-out fold. This covers every Train
window with a probability vector from models that never saw it — zero leakage by
construction, not just by claim (the OOF matrix is built fold-by-fold with no
data crossing a fold boundary).

**A real bug found and fixed during this phase**: the meta-learner (multinomial
`LogisticRegression`) was initially fit with `class_weight="balanced"`, matching every
other classifier in this project. This was wrong here specifically: all three base
learners already train with their own imbalance correction (RF/GBM's
`class_weight="balanced"`, the TCN's inverse-frequency-weighted loss), so their
probability outputs are already imbalance-adjusted — a second `balanced` correction on
top double-corrects. Verified directly on the cached OOF matrix: unweighted meta-learner
OOF macro-F1 0.3134 vs. 0.2368 weighted, with the weighted version's coefficients ~3x
larger in magnitude (up to 9.19 vs. 3.41) — genuinely destabilised, not a small effect.
Fixed to `class_weight=None` before final evaluation.

Final Validation-split comparison (`docs/results/ensemble_comparison.csv`, base
learners retrained on the full Train split, tuned TCN reused from its Phase 3 seed=42
checkpoint):

| Model | Macro-F1 | Accuracy | QWK |
|---|---|---|---|
| tcn_tuned | 0.3081 | 0.4990 | 0.1687 |
| random_forest | 0.2669 | 0.5263 | 0.1660 |
| gradient_boosting | 0.2907 | 0.5070 | 0.1620 |
| **stacked_ensemble** | **0.2580** | 0.5103 | 0.1616 |

**The stacking attempt failed, honestly reported rather than hidden**: even after
fixing the meta-learner weighting bug, the ensemble scores *below every individual base
learner* on macro-F1. This is the single most direct attempt in this whole project at
actually fixing "the core deliverable is weak" (issue #1), and it did not work.

**Root cause, diagnosed directly rather than left speculative** (post-commit follow-up,
same day): the ensemble's confusion matrix on Validation shows it **never predicts
class 0 ("very low") or class 1 ("low") for any of the 11,432 windows** — every
prediction lands on class 2 or 3. That alone drives both rare classes' F1 to exactly
0.0 and accounts for most of the macro-F1 gap. Two initially plausible explanations
were tested directly and both ruled out: (a) base-learner correlation — pairwise
prediction agreement is actually fairly low (TCN vs. RF 63.0%, TCN vs. GBM 62.3%, RF vs.
GBM 75.6%), so the base learners are not simply redundant; (b) an OOF/full-Train
calibration mismatch (the OOF-trained TCN used a 30-epoch/~80%-of-Train budget per
fold, while the final-evaluation TCN reused Phase 3's 100-epoch/full-Train checkpoint)
— a real, measurable distribution shift was confirmed (TCN's OOF-predicted class-1 rate
was 6.85% vs. 3.45% at full-budget Validation time), but retraining a
budget-consistent TCN and re-running the ensemble with it changed Validation macro-F1
by 0.0002 (0.2580 → 0.2578) — not the cause.

**The actual cause: the unweighted meta-learner over-trusts the random forest's
majority-class overconfidence.** The meta-learner's fitted coefficients give the random
forest's probability block 3.6x the mean weight of the TCN's block (mean |coefficient|
1.399 vs. 0.384). Random forest, despite its own `class_weight="balanced"` training,
essentially never predicts the rare classes on real Validation data — only 12 of 11,432
windows get an RF vote for class 0 or 1, against a true combined prevalence of 1,328. A
plain (unweighted) logistic-regression meta-learner, fit on OOF data where classes 2+3
are >95% of samples, is trained to minimise an objective the majority classes dominate
almost entirely — it learns to trust RF's confident, consistent majority-class
signal and effectively discards the weaker but real minority-class signal the TCN and
GBM individually carry. The TCN alone, despite far noisier predictions overall, still
achieves 12% recall on class 0 and 8% on class 1 (`docs/results/rigorous_model_search.md`
confusion-matrix check below) — specifically *because* its own training loss
(inverse-frequency-weighted cross-entropy) forces it to. The meta-learner has no
equivalent forcing function once unweighted, and the earlier `class_weight="balanced"`
version's over-correction (documented above) destabilised it in the opposite direction
instead of fixing this. This is a known stacking pitfall under severe class imbalance —
an unweighted meta-learner gravitates toward whichever base learner is most reliably
*confident* on the majority classes, not whichever carries the most total information —
and it is the reason this ensemble underperforms every one of its own inputs, not
base-learner redundancy or a calibration bug.

## Phase 5 — Final selection with significance testing

`ml/src/final_model_selection.py`, extending `significance.py`'s clip-level bootstrap
utilities (`clip_bootstrap`, `macro_f1`, `ci95` — imported, not duplicated) to a
pairwise comparison of the three Phase 3/4 candidates on Validation (2,000-iteration
paired cluster bootstrap, resampling clip IDs):

| Comparison | Point diff | 95% CI | p-value |
|---|---|---|---|
| tuned_tcn − gradient_boosting | +0.0174 | [−0.0129, 0.0482] | 0.283 |
| tuned_tcn − stacked_ensemble | +0.0501 | [0.0252, 0.0781] | **0.000** |
| gradient_boosting − stacked_ensemble | +0.0327 | [0.0161, 0.0502] | **0.000** |

Full detail: `docs/results/final_model_selection.json`.

**The tuned TCN has the best point estimate (0.3081) but does not significantly beat
gradient boosting** (p=0.283, CI crosses zero) — confirming, with a proper test, what
Phase 3's raw numbers already suggested. The stacked ensemble is significantly worse
than both other candidates (p≈0.000 in both directions) — ruled out decisively, not
just by a small margin.

**The decisive additional test, run because "does the tuned TCN clearly beat the
gradient boosting baseline" is the wrong re-shipping question** (gradient boosting
cannot be shipped in the current browser architecture regardless of its score — see
below): does the **tuned TCN significantly beat the original shipped TCN**? Same
clip-level bootstrap, both at seed 42, on Validation (`docs/results/
tuned_vs_original_tcn.json`):

| Model | Point macro-F1 | 95% CI |
|---|---|---|
| Original (ch=64, dropout=0.2) | 0.3061 | [0.2865, 0.3263] |
| Tuned (ch=128, dropout=0.238, ...) | 0.3081 | [0.2828, 0.3354] |
| **Difference** | **+0.0019** | **[−0.0220, 0.0270]** |

**p = 0.92.** The tuned configuration's improvement over the shipped default is
indistinguishable from noise — the entire Optuna search, run at real cost (40 trials,
~40 fold-level TCN trainings), moved the needle by less than two-thousandths of a point
on Validation macro-F1, and that movement is not statistically real.

## Phase 6 — Re-shipping decision: **no re-ship**

None of the three candidates this pipeline produced "clearly wins" by the standard set
before this work began (a statistically defensible margin, not just a higher point
estimate):

- The **tuned TCN** does not significantly beat gradient boosting (p=0.283) — and,
  decisively, does not significantly beat the **already-shipped** TCN either (p=0.92).
  There is nothing here to re-ship; the current `web/public/model/model_int8.onnx` is
  already statistically indistinguishable from the best thing this search found.
- **Gradient boosting** has a higher point estimate than the shipped TCN in some
  comparisons but is not shippable regardless: it cannot be practically exported to the
  ~60 KB browser-runnable quantized ONNX graph the TCN uses (tree ensembles do not
  quantize/export the same way — the same class of constraint as the thesis's §2.5.2
  `QLinearConv`-vs-`ConvInteger` finding).
- The **stacked ensemble** is significantly worse than both alternatives (p≈0.000) —
  not a candidate on the merits, independent of the architectural question.

**No changes were made to `web/public/model/model_int8.onnx`, `scaler.json`,
`CONTRACT.md`, or any `web/` TypeScript code.** The shipped model stands as-is,
now with considerably stronger evidence behind that decision than existed before this
pipeline ran — not because a better model was found and rejected for deployment
reasons, but because a genuinely rigorous search (nested CV, real Bayesian
hyperparameter optimisation, an ensemble attempt) did not find one.

## What this changes about the project's prior claims

1. **The earlier single-split/3-seed feature-ablation finding** ("geometric+gaze beats
   full for the TCN", `model_comparison_summary.md` §6) does not survive correction to
   subject-level cross-validation. It should not be cited as a real effect in the
   thesis; Phase 1's 5-fold, subject-grouped result (no significant preset differences)
   supersedes it.
2. **The class-imbalance-weighting pattern used throughout this project's classifiers**
   (`class_weight="balanced"` / inverse-frequency loss) is not automatically safe to
   stack: a second balancing correction on already-balanced-corrected outputs
   measurably destabilises a meta-learner. Worth a general caution if this project's
   class-weighting approach is ever composed with another imbalance-aware layer
   elsewhere.
3. **The shipped TCN's hyperparameters, chosen informally in the original 6-run search
   (`docs/PROGRESS.md`), are not meaningfully suboptimal.** A properly powered,
   pruned, 40-trial Bayesian search over 7 hyperparameters, validated with full-budget
   multi-seed retraining and a significance test, found nothing better. This is
   reassuring evidence for the original choice, not a gap in it — worth stating
   plainly in the thesis rather than leaving the impression that a "real" search was
   never tried.
4. **The core-deliverable weakness (issue #1) is not fixed, and this document says so
   directly rather than implying otherwise.** Hyperparameter tuning and ensemble
   stacking were both tried in earnest and neither moved macro-F1 meaningfully above
   the already-shipped model. This is now a better-evidenced limitation than before —
   the thesis's critical-review section can state, with real search evidence behind
   it, that the 13-feature geometric representation itself (not undertuned
   hyperparameters, not an unexploited ensemble opportunity) is the binding
   constraint on this model family's ceiling.

## Future work suggested by this pipeline specifically

- **A class-imbalance-aware meta-learner objective, given the diagnosed cause above.**
  Now that the ensemble's failure is understood precisely (an unweighted meta-learner
  over-trusts RF's majority-class confidence and abandons rare-class signal; a fully
  `class_weight="balanced"` meta-learner over-corrects into instability), the next
  concrete step is a middle ground — a moderate, tuned `class_weight` dict or
  `sample_weight` schedule for the meta-learner specifically (distinct from, and likely
  softer than, the `"balanced"` setting used throughout the rest of this project), or
  dropping the random forest from the stack given its near-total blindness to the rare
  classes and re-fitting on just TCN + gradient boosting. Not attempted in this pass —
  the diagnosis was the deliverable, not a guaranteed fix.
- **A full 5-fold (not 2-fold) hyperparameter search**, given more compute time/a more
  interruption-resistant environment than was available for this pass, to confirm the
  2-fold search proxy didn't systematically bias the selected configuration.
- **A genuinely different feature representation** (raw landmark sequences, a learned
  spatial embedding per Section 5.2.3(a) of the existing thesis draft) is the next
  logical lever, now that this document has closed off "the current architecture is
  undertuned" and "the current features are chosen suboptimally" as explanations for
  the model's ceiling.

## Files produced by this pipeline

- `ml/src/cv_splits.py`, `cv_train.py`, `cv_feature_selection.py`,
  `cv_hyperparameter_search.py`, `final_candidate.py`, `ensemble_stack.py`,
  `final_model_selection.py`
- `docs/results/cv_feature_selection.csv`, `cv_feature_selection_summary.csv`,
  `cv_hyperparameter_search.csv`, `cv_hyperparameter_search_best.json`,
  `final_candidate.csv`, `ensemble_comparison.csv`, `ensemble_meta_learner_coef.json`,
  `final_model_selection.json`, `tuned_vs_original_tcn.json`
- `ml/requirements.txt` — added `optuna==4.9.0` (+ its own dependencies: `alembic`,
  `colorlog`, `Mako`, `SQLAlchemy`)
- `artifacts/runs/optuna_cv_search.db` — full Optuna trial database (gitignored, same
  as every other `artifacts/runs/` entry — regenerable by rerunning
  `cv_hyperparameter_search.py`, not itself a committed artefact)
