# Model comparison, ablation, and significance testing — findings

Extends `docs/results/baselines.csv` and the TCN evaluation (`metrics_validation.csv`,
`metrics_test.csv`) with five additions: a gradient-boosting classical baseline,
quadratic-weighted kappa (QWK, an ordinal-aware metric), an LSTM/GRU architecture
comparison, a feature-family ablation, and clip-level bootstrap significance testing.
This closes a gap identified in a masters-level review of this project: the thesis
draft (`docs/thesis/FYP_Report.md`) trains and reports only one deep architecture (the
TCN) against two classical sanity-check baselines, with no statistical test for whether
the TCN's reported advantage is real, and no empirical evidence for the "three visual
modality families" framing beyond the architecture that assumes it. All numbers below
are copied verbatim from the CSV/JSON files named; none are re-derived or rounded by
hand. This file does not edit `docs/thesis/FYP_Report.md` — it is a standalone record,
in the same spirit as `docs/privacy.md` and `docs/demo-failure-modes.md`.

All new work is Validation-split only, except the significance test, which reads the
already-frozen Test-split predictions from the TCN checkpoint that produced the
committed `metrics_test.csv` (`artifacts/runs/20260801_185630`) — Test is not
re-evaluated against any newly trained model.

## 1. Classical baselines, now including gradient boosting

Source: `docs/results/baselines.csv`.

| Split | Model | Macro-F1 | Accuracy | QWK |
|---|---|---|---|---|
| Validation | majority | 0.1813 | 0.5689 | 0.0 |
| Validation | logreg | 0.2420 | 0.3109 | 0.1387 |
| Validation | random_forest | 0.2669 | 0.5263 | 0.1660 |
| Validation | **gradient_boosting** | **0.2907** | 0.5070 | 0.1620 |
| Validation | TCN (`metrics_validation.csv`) | 0.3043 | 0.4433 | 0.1330 |
| Test | majority | 0.1655 | 0.4948 | 0.0 |
| Test | logreg | 0.2479 | 0.3540 | 0.0806 |
| Test | random_forest | 0.2643 | 0.5054 | 0.0961 |
| Test | **gradient_boosting** | **0.2910** | 0.4976 | 0.1138 |
| Test | TCN (`metrics_test.csv`) | 0.2475 | 0.3686 | n/a¹ |

¹ QWK was added to `eval.py` after the Test split was consumed (2026-08-02); it is not
retroactively computed for `metrics_test.csv`, consistent with the project's "Test
touched exactly once" rule.

**Gradient boosting (`HistGradientBoostingClassifier`, `sample_weight` from
`compute_sample_weight("balanced", ...)`) is now the strongest single model on the Test
split, classical or not: macro-F1 0.2910 against the TCN's 0.2475** — a larger margin
than random forest's already-reported win (0.2643). On Validation the TCN is still
ahead (0.3043–0.3061 depending on checkpoint vs. 0.2907), so the ordering is
split-dependent; Section 4 below tests whether that Validation lead is even
statistically distinguishable from noise.

## 2. Architecture comparison — TCN vs. LSTM vs. GRU

Source: `docs/results/architecture_comparison.csv`. All three trained at identical
hyperparameters (`train.py` defaults: weighted CE, Adam lr 1e-3, seed 42,
state-loss-weight 0.5), Validation split.

| Arch | Parameters | Macro-F1 | Accuracy | QWK |
|---|---|---|---|---|
| tcn | 41,544 | **0.3061** | 0.4354 | 0.1304 |
| lstm | 20,744 | 0.2823 | 0.3652 | 0.1393 |
| gru | 15,688 | 0.2837 | 0.3982 | 0.1390 |

The TCN beats both recurrent alternatives on macro-F1 by a clear margin (≈2.2–2.4
points), despite having roughly double the parameters of the LSTM and 2.6× the GRU —
so the gap isn't simply "the TCN is bigger." This is consistent with the direction (if
not the magnitude) of Abedi and Khan's (2021) own finding that a TCN temporal head beats
an LSTM one on this same dataset, now confirmed on this project's own 13-feature
geometric representation rather than only cited from prior work on a pixel-based one.
LSTM/GRU QWK is marginally *higher* than the TCN's despite lower macro-F1 — both
recurrent models lean harder on the majority classes (lower per-class balance, higher
QWK, which rewards keeping errors "close" more than rewarding rare-class recall).

These two architectures were trained once each, not swept — the comparison is at
matched hyperparameters, not each architecture's own best case.

## 3. Feature-family ablation

Source: `docs/results/feature_ablation.csv`. Four configs (`feature_groups.py`):
geometric-only (8 features: EAR×3, MAR, brow×2, face_area, face_present),
+pose (11: adds yaw/pitch/roll), +gaze (10: adds gaze_x/gaze_y instead of pose),
full (13, all three families). Validation split.

| Config | n | logreg | random_forest | gradient_boosting | tcn |
|---|---|---|---|---|---|
| geometric | 8 | 0.2680 | 0.2564 | 0.2801 | 0.2911 |
| geometric+pose | 11 | 0.2412 | 0.2528 | 0.2844 | 0.3014 |
| geometric+gaze | 10 | 0.2640 | 0.2589 | 0.2895 | **0.3252** |
| full (all 3) | 13 | 0.2427 | 0.2669 | **0.2907** | 0.3061 |

**Mixed evidence for the "three modality families" framing, reported honestly rather
than smoothed over:**

- **Gradient boosting and (mostly) random forest show the expected pattern**: each
  added family raises macro-F1, full beats every subset. This is direct empirical
  support for the multimodal framing — for these two models.
- **The TCN and logistic regression do not show a monotonic pattern.** For the TCN,
  geometric+gaze (0.3252) *beats* the full 13-feature set (0.3061) — adding pose on top
  of geometric+gaze would, by this single run, cost accuracy rather than add it. For
  logreg, geometric-only (0.2680) beats every other subset including full (0.2427).
- **Caveat, stated plainly**: each cell is a single training run, not averaged over
  seeds or cross-validated — TCN training is seed-42-only and known from the original
  6-run hyperparameter search (`docs/PROGRESS.md`) to have run-to-run variance on this
  scale of dataset. The geometric+gaze TCN result (0.3252) is the single best macro-F1
  in this entire study and is flagged here as a genuine, interesting result **and** a
  candidate for a multi-seed follow-up before treating "drop pose" as a real
  recommendation, not an artifact of one run's variance — consistent with this
  project's own standard of not overclaiming from a single number (`FYP_Report.md`
  §5.2.3(b) makes the identical point about class-imbalance remedies).
- What is not in question: geometric-only is the weakest TCN config (0.2911) and every
  addition (pose, gaze, or both) beats it — some signal from pose and gaze exists, the
  ambiguity is specifically about whether they are additive together or redundant/
  conflicting once combined.

## 4. Clip-level bootstrap significance testing

Source: `docs/results/significance.json`. Method: 2,000-iteration paired cluster
bootstrap, resampling **clip IDs** (not windows) with replacement — windows from the
same clip share up to 20 of their 30 frames (stride-10 windowing) and are not
independent samples, so resampling at the window level would understate the true
uncertainty. TCN vs. the best classical baseline (gradient boosting, per §1) on each
split:

| Split | TCN macro-F1 (95% CI) | Gradient boosting macro-F1 (95% CI) | TCN − GBM (95% CI) | p-value |
|---|---|---|---|---|
| Validation | 0.3061 (0.2865–0.3263) | 0.2907 (0.2720–0.3104) | +0.0154 (−0.0071–0.0381) | 0.194 |
| Test | 0.2475 (0.2330–0.2613) | 0.2910 (0.2772–0.3057) | −0.0435 (−0.0594 – −0.0280) | 0.000 |

**On Validation, the TCN's lead over gradient boosting is not statistically
significant** — the 95% CI for the difference crosses zero (p=0.194). The
hyperparameter search that selected the TCN's final configuration (`docs/PROGRESS.md`,
6 runs) picked the best of several close configurations on this same margin; this
result says that margin is not distinguishable from noise at conventional confidence
levels, once clip-level clustering is accounted for.

**On Test, the reverse gap is statistically significant** — gradient boosting is
robustly better (p≈0.000, CI entirely below zero across all 2,000 resamples). This is
the same crossover the thesis already reports honestly (§4.3.3, "the gap narrows
somewhat on Test") but not previously quantified: it is not a small, possibly-noisy
reversal, it is the single most confident result in this whole comparison.

## 5. What this means for the thesis

This does **not** mean the TCN was the wrong architecture to ship. Gradient boosting
and random forest operate on a 65-dimensional hand-aggregated statistic of each window
(mean/std/min/max/range per feature) and are not straightforwardly exportable to a
60 KB, browser-runnable, quantized ONNX graph the way the TCN is (§2.5.2's `QLinearConv`
vs. `ConvInteger` finding is architecture-specific to a neural network graph, not a
tree ensemble) — the edge-deployment constraint that motivated the TCN in the first
place is not something this comparison relaxes. The TCN also convincingly beats the two
other deep-learning alternatives that share its deployability profile (§2 above).

What this data **does** mean: the thesis's current framing — "the temporal model beats
both classical baselines on validation macro-F1... providing direct, measured evidence
that the additional temporal structure a TCN exploits... is worth the added deployment
complexity" (§4.3.3) — overstates the case as written. That validation-set advantage is
not statistically significant (§4 above), a stronger classical baseline than either
tested at the time (gradient boosting) exists and beats the TCN outright on Test, and
the feature-family evidence for the model's own "multimodal" framing is genuinely mixed
rather than uniformly supportive. A future revision of §4.2–4.3.3 and §5.2 should
incorporate all five findings above — they strengthen the report's critical-analysis
depth considerably (this is exactly the kind of result a masters-level "critical
review" should surface, not omit), and they do not require walking back the deployment
argument, only the predictive-superiority claim.

---

# Phase 2 — multi-seed robustness, further models, hyperparameter search

Extends Sections 1–5 above with three additions requested after reviewing those
results: every Phase-1 architecture/ablation number was a **single training run per
config** (seed 42 only); this phase re-runs the contested ones at 2 additional seeds
(7, 123) to separate real effects from seed variance, adds a Transformer architecture
and a CORAL ordinal-regression variant of the TCN, and runs a lightweight
hyperparameter search. All new work is Validation-split only, same discipline as
Phase 1. `docs/thesis/FYP_Report.md` is still not edited in this pass.

## 6. Multi-seed robustness

Source: `docs/results/multi_seed_robustness.csv` (long format: experiment, config,
seed, macro_f1, accuracy, qwk — 24 rows, 3 seeds × 8 configs).

| Group | Config | Mean macro-F1 (n=3) | Std | Min | Max |
|---|---|---|---|---|---|
| architecture | tcn | 0.3015 | 0.0049 | 0.2964 | 0.3061 |
| architecture | lstm | 0.2938 | 0.0139 | 0.2823 | 0.3092 |
| architecture | gru | 0.2907 | 0.0064 | 0.2837 | 0.2964 |
| architecture | transformer | 0.3019 | 0.0066 | 0.2956 | 0.3087 |
| feature_ablation | geometric | 0.3003 | 0.0084 | 0.2911 | 0.3075 |
| feature_ablation | geometric_pose | 0.2953 | 0.0053 | 0.2915 | 0.3014 |
| feature_ablation | geometric_gaze | 0.3152 | 0.0089 | 0.3083 | 0.3252 |
| feature_ablation | full | 0.3015 | 0.0049 | 0.2964 | 0.3061 |

**Architecture comparison does NOT survive multi-seed averaging as previously
stated.** At seed 42 alone, TCN (0.3061) clearly led Transformer (0.3014), LSTM
(0.2823), GRU (0.2837) — Section 2's original conclusion, drawn from one seed each.
Averaged over 3 seeds, TCN (mean 0.3015) and Transformer (mean 0.3019) are
statistically indistinguishable — the "TCN beats the other sequence-model families"
claim only holds cleanly against LSTM and GRU (means 0.007–0.011 below TCN's), and
even there, individual runs overlap (LSTM's own seed-7 run, 0.3092, beat every one of
TCN's three seeds). This is exactly the kind of correction multi-seed testing exists
to catch: a single favourable seed made the seed-42-only comparison look more decisive
than it is.

**The geometric+gaze-beats-full TCN ablation finding DOES survive multi-seed
averaging, and is now the strongest-evidenced result in this entire study.**
geometric+gaze's mean (0.3152, std 0.0089) exceeds full's mean (0.3015, std 0.0049) by
0.0137 — larger than either config's own std, and geometric+gaze's *worst* seed
(0.3083) still beats full's *best* seed (0.3061), i.e. the two configs' seed ranges do
not overlap at all. This is no longer a single-run curiosity flagged for follow-up; it
is a reproducible pattern, with a plausible mechanism: for the TCN specifically, pose
features (yaw/pitch/roll) measurably hurt validation macro-F1 relative to dropping
them (geometric_pose mean 0.2953 < geometric-alone mean 0.3003), while gaze features
clearly help (geometric_gaze mean 0.3152, the best of all four configs, on every
seed). The "three visual modality families" framing (CONTRACT.md §8) is directly
falsified for the TCN as tested: adding all three families does not beat two of them.

## 7. Ordinal regression (CORAL)

Source: `docs/results/ordinal_comparison.csv`. CORAL (Cao, Mirjalili and Raschka,
2020; implemented in `ml/src/model.py`'s `CoralLayer`/`coral_loss`/`coral_predict`,
wired via `train.py --ordinal`) replaces the TCN's 4-way softmax engagement head with
a rank-consistent ordinal formulation, directly optimising the label's ordinal
structure rather than only measuring it after the fact via QWK.

| Model | Macro-F1 | Accuracy | QWK |
|---|---|---|---|
| tcn_softmax (nominal, existing) | 0.3061 | 0.4354 | 0.1304 |
| tcn_coral (ordinal, new) | 0.2571 | 0.4745 | 0.1558 |

**A genuine trade-off, not a strict improvement or regression.** CORAL trades
macro-F1 for accuracy and QWK: −0.049 macro-F1, but +0.039 accuracy and +0.025 QWK.
This is consistent with what CORAL's rank-consistent decoding predicts: by
construction it concentrates predictions closer to the ordinal middle of the
distribution (fewer wild, far-off errors — hence higher QWK/accuracy) at the direct
cost of rare-class recall (hence lower macro-F1, the metric this project has used
throughout specifically *because* it weighs the rare classes equally — thesis Section
1.5). Whether this trade is worth taking depends on which error type matters more for
the deployed use case; by this project's own already-adopted metric (macro-F1
primary), CORAL is a worse model, not a free win — but the accuracy/QWK improvement is
real and would matter more under a different design goal.

## 8. Hyperparameter search

### Gradient boosting (`RandomizedSearchCV`)
Source: `docs/results/hyperparameter_search.csv`.

| Model | Macro-F1 | Accuracy | QWK |
|---|---|---|---|
| gbm_default (sklearn defaults) | 0.2907 | 0.5070 | 0.1620 |
| gbm_randomsearch_best (20-iter, 3-fold CV, scoring=f1_macro) | 0.2717 | 0.3680 | 0.1359 |

**The search's own "best" configuration is worse on Validation than plain
defaults.** `{max_iter: 200, max_depth: 3, learning_rate: 0.01, l2_regularization:
0.1}` scored highest under 3-fold cross-validated macro-F1 on Train, but scores 0.019
*lower* on the actual held-out Validation split than doing no search at all. This is a
real, reportable finding about the search itself, not a bug: with 42,856 training
windows split further into 3 CV folds, and macro-F1 dominated by rare-class behaviour
(thesis Section 1.5), the CV estimate is noisy enough that "best on 3-fold Train CV"
and "best on Validation" diverge. Plain sklearn defaults remain the reported
gradient-boosting baseline throughout this project (Sections 1–5) — this result is a
reason not to switch, not a missed improvement.

### TCN channel/dropout grid
Source: `docs/results/tcn_grid.csv`. Single run per config, seed 42, full feature set.

| Channels | Dropout | Parameters | Macro-F1 | Accuracy | QWK |
|---|---|---|---|---|---|
| 64 (shipped) | 0.2 (shipped) | 41,544 | **0.3061** | 0.4354 | 0.1304 |
| 32 | 0.2 | 11,560 | 0.2953 | 0.3988 | 0.1637 |
| 96 | 0.2 | 89,960 | 0.3012 | 0.4310 | 0.1353 |
| 64 | 0.4 | 41,544 | 0.3001 | 0.4177 | 0.1244 |

**The shipped configuration (64 channels, 0.2 dropout) is the best of the four
tested, by a small margin.** Neither narrower (32ch, −73% params), wider (96ch, +116%
params), nor higher-dropout alternatives beat it on Validation macro-F1. This is a
small, targeted grid (4 points, not a real search), so it does not prove 64/0.2 is
optimal — only that it is not obviously beaten by the nearby alternatives tested.
Worth noting alongside the edge-deployment framing (thesis Section 2.3.2): the
89,960-parameter (96-channel) variant is still under the project's self-imposed 100k
budget and does not even win, mild further evidence that this task's ceiling is closer
to a data/feature-representation limit than a TCN-capacity limit at these widths.

## 9. Synthesis — what Phase 2 changes about Phase 1's conclusions

Multi-seed testing was the right thing to demand: one Phase-1 conclusion (TCN clearly
beats LSTM/GRU/Transformer) weakens under it, and one (geometric+gaze beats full for
the TCN) strengthens into the best-evidenced finding in the whole study.

- **Weakened**: "the TCN beats the other architecture families" now only holds
  against LSTM/GRU on average, not against the Transformer (statistically tied,
  0.3015 vs 0.3019 mean macro-F1) — Section 2's single-seed table overstated this.
- **Strengthened**: the geometric+gaze-over-full TCN ablation result is no longer a
  "flagged for confirmation" single run — it is reproducible across 3 non-overlapping
  seed ranges, with a specific, plausible mechanism (pose hurts, gaze helps), not an
  unexplained anomaly.
- **New, contained findings**: CORAL ordinal regression is a real macro-F1-for-
  accuracy/QWK trade-off, not a strict win; a lightweight hyperparameter search did
  not improve on either the gradient-boosting baseline or the shipped TCN
  configuration, and the search itself produced a cautionary result about CV-based
  search reliability at this dataset size.

None of this changes the deployment argument for the TCN (Section 5) — it remains the
only one of these architectures actually shipped, quantized, and running in-browser.
What it does mean, on top of Phase 1's already-stated correction: a future revision of
the thesis's Section 2.3.2/4.3 "multimodal = three families" framing should be
qualified specifically (gaze helps, pose does not, for the TCN) rather than presented
as uniformly supported, and any claim that the TCN's architecture choice beats other
sequence-model families should cite the multi-seed mean, not the seed-42 point
estimate. As with Phase 1, this file states these findings and stops —
`docs/thesis/FYP_Report.md` is not edited in this pass either.
