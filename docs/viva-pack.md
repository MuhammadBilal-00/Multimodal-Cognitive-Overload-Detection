# Viva pack — narrative, timing, Q&A bank, contingencies

Companion to `docs/dry-run-checklist.md` (the live demo script). This file
is the *spoken-defence* preparation: the story to tell, the questions a
strict panel will ask, and the answer each one deserves. Every answer here
is backed by a committed artefact — cite it out loud; it is the difference
between "I believe" and "we measured".

## 1. The three-sentence story (open with this)

"We built a system that watches a learner's engagement entirely inside
their own browser — no frame, feature, or prediction ever leaves the
machine, and we don't ask you to trust that: we recorded it, blocked a
real third-party telemetry call we caught in the act, and made the
verification re-runnable in one command. The model itself is deliberately
tiny — 60 KB — and we then attacked our own results harder than any
reviewer would: stronger baselines, four alternative architectures, a
40-trial hyperparameter search, an ensemble, and a full statistical audit,
which together proved the shipped model sits at its feature
representation's honest ceiling. Two real defects our own verification
caught along the way — and the corrections they forced — are, we'd argue,
the strongest evidence for the methodology this project is really about."

## 2. Timing budget (adjust to the actual slot)

| Segment | Minutes |
|---|---|
| Story + architecture (one diagram) | 3 |
| Live demo: normal operation → benchmark button → 2–3 failure modes on purpose | 6 |
| Privacy evidence (privacy_trace.json on screen, telemetry-block lines) | 2 |
| Results honestly: clip-level table, the ceiling argument, one negative result | 4 |
| Reserve for questions/overrun | rest |

Rules: never let the demo exceed its slot — the benchmark button is the
natural cut point; trigger failure modes yourself before the panel does.

## 3. Q&A bank

**Q1. Your accuracy is only ~44%. Why should anyone care?**
Published SOTA on this 4-class task is 63.9% accuracy — from a GPU
ResNet processing raw video that cannot run privately in a browser
(Abedi & Khan 2021, cited §2.3). Random guessing is 25%; the majority
class alone gets ~50% while being useless (macro-F1 0.166). The labels
are noisy crowd annotations of subtle distinctions, which caps what
*anyone* can measure. Our contribution is the privacy/deployability trade
made honestly and quantified exactly — and the app is designed around
what 44% supports: trends over time, never point judgements about a
student. (Thesis §1.5, §4.3.3, §5.1.)

**Q2. Gradient boosting beats your TCN on Test — why did you ship the TCN?**
Four-part answer, in order: statistically level on Validation (p=0.194);
no browser path exists for a tree ensemble (the deployment constraint is
the premise); the TCN beats LSTM/GRU at matched budgets and a 40-trial
Bayesian search couldn't improve it (p=0.92); therefore the binding
constraint is the 13-feature representation, which is exactly future-work
item 4. Files: `significance.json`, `rigorous_model_search.md`. Never
deny the GBM result — we published it ourselves.

**Q3. Figure 4.3 shows below-chance AUC for class 0. Your model is worse than random?**
No — 32 windows from 4 clips cannot estimate an AUC; a value below 0.5 on
that sample means "unmeasurable", not "anti-predictive". That is why the
3-class-merged metric exists. Class 2's ~0.5 is different: one-vs-rest
scoring is structurally hard for the middle band of an ordinal scale —
which is why macro-F1 over hard assignments is primary. (§4.3.2.)

**Q4. Which checkpoint produced which numbers? Your validation and test artefacts differ.**
Deliberate and disclosed (§4.3.2): Test was consumed exactly once
(2026-08-02, frozen checkpoint); the later states-head retrain shipped to
the browser was never re-evaluated on Test, protecting the once-only
claim. Validation artefacts describe the shipped checkpoint.

**Q5. You report your parity gate proudly — but didn't it miss a bug for weeks?**
Yes, and we tell that story on purpose (§4.1, Amendment 4): the brow
formula divergence hid inside a passing gate because the tolerance was
wide enough for real runtime noise. Our own audit caught it — via the
principle that a feature-*selective* residual can't be noise — fixed it,
and the gate's worst-case diff dropped 0.0157→0.0035. Verification
catching our own mistakes twice is the thesis's actual argument.

**Q6. Is any of this actually enforced in CI?**
Precisely answered in §3.4: unit/contract suites (38 Python + 41 TS +
typecheck) and the export-parity gate run on every push; the J1 parity
gate runs on any runner seeded with the licence-restricted DAiSEE fixture
and skips with a visible warning otherwise; browser smoke and e2e are
release-point scripts for the same licensing reason.

**Q7. How do I know your privacy claim isn't just a README sentence?**
`ml/scripts/privacy_trace.py` — run it now if you like: production build,
75 s, full CSP enforced; 39 requests all same-origin, zero external, the
MediaPipe telemetry POST blocked twice on record, and the script *fails*
on any deviation. Plus the header set itself: default-src lockdown,
Permissions-Policy camera=(self). (§4.4, `privacy_trace.json`.)

**Q8. Why only 13 hand-crafted features instead of deep learning on pixels?**
Three reasons in the thesis (§2.1.2): browser compute budget; a formula
can be implemented twice and *numerically verified* across languages (a
CNN embedding cannot); and interpretability. The measured cost is the gap
to SOTA — quantified, §4.3.3 — and the search phase proved the ceiling is
this representation, not our tuning of it.

**Q9. What did the AI tools do, and what did you do?**
Answer from the Declaration of AI Use, honestly: AI assisted code
drafting/refactoring, experiment execution and logging, and report
drafting under the author's review; every result comes from committed
code executed on the real dataset; design decisions and the contract with
the Track B partner are the author's; and being able to answer every
question in this pack unaided is the demonstration of ownership. Do not
be defensive; be fluent.

**Q10. Your dataset numbers changed between drafts — 9,032 vs 9,067 vs 8,570?**
A real bookkeeping error we found and fixed publicly (§1.5): a resumed
extraction run's stats file only covered its own pass (9,032); 9,067
clips exist on disk with feature CSVs; the label∩extraction intersection
actually trained on is 8,570. The correction is in the abstract's history
and the limitation is owned in text.

**Q11. What's the states panel actually good for at AUC 0.56–0.64?**
Trend indication, and the UI says so: independent likelihoods, don't sum
to 100%, OOD notices when input leaves the training distribution. We
report the head's full table (§4.3.2) because shipping an output without
metrics would be an omission.

**Q12. What would you do with three more months?**
Future-work list, in evidence order (§5.3): ship the four-constant
threshold calibration to the browser (+14% relative Test macro-F1,
already validated); the two remaining benchmark machines; the hybrid
learned-representation study — now *justified* by the search having
eliminated tuning as the ceiling; an aperture-gated gaze_y redefinition
(the audit's 26σ blink finding); a larger low-engagement evaluation set.

**Q13. Two people in frame — whose engagement is shown?**
Honest limitation (§5.2.2c): the overlay's highlighted "primary" face and
the model-feeding single-face detector can disagree; they are separate by
parity necessity (Amendment 2). Single-learner use is the design target;
multi-learner would need reconciliation.

**Q14. Why 3-second windows / 10 Hz?**
Frozen contract §6: 30 frames at 10 Hz = the training window; the live
cadence shows one prediction per non-overlapping window for readability
(Amendment 1). And say it before they find it: the sampling loop
originally drifted to ~8.6 Hz on 60 Hz displays — found in audit, fixed
with an accumulator clock, disclosed in §5.2.2b.

## 4. Contingencies

| Risk | Plan |
|---|---|
| Camera claimed by Teams/Zoom (`NotReadableError`) | The app shows a specific message for it (WebcamFeed handles 6 DOMException types — demo-worthy in itself). Close the other app; if stuck, switch to the backup browser profile. |
| No network in the room | Everything is self-hosted; say so and point at the privacy story. Verify once beforehand with WiFi off. |
| Projector at odd resolution | UI is responsive; if the overlay misaligns, that's `coverFit` handling 16:9 — zoom the browser, don't resize the camera. |
| App won't reach "Live" | Fallback: `docs/results/privacy_trace.json` + `app_e2e.json` + the synthetic screenshot prove the current build works; walk the evidence instead. |
| Hostile "this is all AI-generated" | Q9 answer + fluency across this pack + the git history of caught-and-corrected mistakes, which no one fakes. |

## 5. Do-not-say list

- "The model beats the classical baselines" (Test says otherwise; Q2).
- "~78% binary accuracy" without its trivial-baseline caveat (88–95%).
- "CSP makes exfiltration impossible" — say "blocks and logs every route
  we could enumerate, verified live" (§4.4's corrected scope).
- Any number you can't name the file for.
