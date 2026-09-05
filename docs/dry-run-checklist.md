# Freeze & dry-run checklist

The guiding principle is *"Know what breaks before the panel finds it."
This is a script for the **live** rehearsal — sitting in
front of an actual webcam, in the actual room if possible, ideally with
someone else watching and taking notes. Everything in
`docs/demo-failure-modes.md` was already verified automatedly (headless
Playwright + real DAiSEE clips fed as a fake camera); this is the
different, necessary step of seeing it happen live, because a live
camera, live lighting, and a live audience surface things headless
automation can't — nervousness-induced fumbling, an unexpected room
light, a laptop that behaves differently than the dev machine.

**This checklist doesn't get to mark Phase 5 done — running it for real
does.**

> **Live rehearsal record — 2026-08-29, production build (commit `63eee66`),
> real webcam.** Part 1 baseline: Live reached; Stats read render 36 fps,
> sampling 10 Hz (the post-fix accumulator rate, verified live), inference
> 3.9 ms, wasm×20, 478 landmarks, model 60 KB; benchmark p50 0.78 ms.
> Part 2 failure modes all as scripted: step-out-of-frame shows the new
> "— no face" suppression (fixed during this very rehearsal — see
> PROGRESS.md); second person raised the People count to 2 with primary
> selection behaving; partial dimming kept tracking and full darkness went
> cleanly to no-face; glasses kept 478 landmarks with gaze populated;
> permission denial produced the specific error message, no stuck spinner.
> First live run of this checklist: **complete.** (Part 3 cross-browser
> applies only if the defense machine differs.)

---

## Before you start

- [ ] Fresh `git pull` on the machine you're actually demoing from — not
      assumed, checked (`git log -1` should show the latest commit).
- [ ] `cd web && npm install && npm run build && npm start` on that exact
      machine — the full clean-install path from `docs/demo-failure-modes.md`
      §1.1, done live, not from cached `node_modules`/`.next`.
- [ ] Camera and lighting checked in the actual room you'll present in —
      not your usual desk setup, if the panel room is different.
- [ ] A second browser (or the same machine's other installed browser)
      ready as a fallback, given the cross-browser findings below.
- [ ] Know your machine's answer to "what happens if the wifi/network
      drops" — it shouldn't matter (everything's self-hosted per
      `docs/privacy.md`), but say that out loud once during rehearsal to
      make sure it's actually true on this machine, not just in theory.

## Part 1 — Normal operation (establish the baseline)

1. Load the app, grant camera permission, let it reach `live` status.
2. Narrate the pipeline stage callouts as they light up: face detected →
   478 landmarks → features panel populates → 3 s window fills → first
   prediction. This *is* the architecture diagram in `docs/architecture.md`,
   just live.
3. Point at the PerfHUD backend/thread readout (`wasm×N`) and state
   plainly that this confirms cross-origin isolation is active — tie it
   to the privacy story (`docs/privacy.md`), not just performance.
4. Run the benchmark panel once ("Run 300 inferences") — this is a
   genuine, reproducible, on-the-spot number, not a slide.

## Part 2 — Deliberately trigger each failure case

Do these **on purpose**, narrating what should happen before it happens,
using the actual observed results from `docs/demo-failure-modes.md` so
you're not guessing live:

| Trigger | Say before | Expected (per `docs/demo-failure-modes.md`) |
|---|---|---|
| Step out of frame | "Watch the readings suppress themselves, not freeze — the model refuses to guess about an empty room" | All 13 features → 0.0000, `face_present=0`; engagement shows "— no face", state bars show em-dashes, trend holds its last real points (no garbage spike), dashboard stays live |
| Bring a second person into frame | "Watch it pick one primary face and dim the other" | Overlay: cyan on primary (larger/closer face), dimmed gray on the other; "People" count increments |
| Cover part of the light / turn off the room light | "Either it keeps tracking or cleanly reports no-detection — never garbage in between" | Verified at two severities: moderate dimming still detects; severe darkness reports clean no-detection |
| Put on glasses (or have someone who wears them step in) | "Iris landmarks are the part most likely to struggle with glasses glare" | Verified clean in testing — full 478-landmark detection, gaze features populated, no degradation observed in that test's lighting |
| Deny camera permission (reload, click "block" this time) | "This should give a clear message, not a stuck spinner" | `WebcamFeed.tsx` shows one of five specific messages depending on the exact failure — confirm the right one appears for an actual denial |

**If any of these behaves differently live than the table says**, that's
exactly the point of rehearsing — note it here, fix or caveat it, re-run
this checklist, don't find out from the panel.

## Part 3 — Cross-browser, if the defense machine might differ

- [ ] If there's any chance the presentation machine/browser isn't the
      one used throughout development, run Part 1 on it at least once
      beforehand. `docs/demo-failure-modes.md` §1.2 has Chrome/Edge/Firefox
      results from this session — Edge is confirmed identical to Chrome;
      Firefox loads and runs but its real-face detection was only checked
      automatedly with a synthetic (non-file-backed) fake camera, not a
      real one — so a live check with a real face in Firefox, if that's
      ever the presentation browser, is genuinely new information, not a
      formality.

## Part 4 — Known, honest limitations to have an answer ready for

Say these before you're asked, based on the project's risk register
philosophy (a properly-analyzed weak result passes review; a hidden one
doesn't):

- Engagement class 0 has only 4 clips in the entire test split — its
  per-class metrics are statistically close to meaningless alone; the
  3-class-merged metric exists specifically because of this
      (the recorded project results).
- Only 1 of the 3 required benchmark machines has been run
  (`docs/benchmarks/README.md` has the runbook for the other two — if
  they're done by defense time, update this line).
- Macro-F1 numbers (0.2475 fp32 / 0.2460 int8 test at window level;
  0.2829 / 44.7% accuracy at the benchmark's clip level with
  validation-calibrated thresholds, `clip_eval_test.json`) are modest in
  absolute terms. **Do NOT claim the model beats the classical baselines
  outright — the panel can check.** The honest, rehearsed answer to "why
  ship the TCN when gradient boosting beats it on Test
  (0.2910 vs 0.2475, p<0.001, `significance.json`)?" is: (1) on
  Validation the two are statistically level (p=0.194); (2) gradient
  boosting cannot ship — there is no browser-WASM path to a 60 KB
  quantized tree ensemble, and edge deployment is the project's premise;
  (3) the TCN beats the recurrent alternatives at matched budgets and a
  40-trial hyperparameter search could not improve it (p=0.92 vs
  shipped) — the configuration is validated at its representation's
  ceiling (`rigorous_model_search.md`); (4) the correct future lever is a
  richer representation, which is future-work item 4 in the thesis.
- If Figure 4.3 (Test ROC) is on screen: class 0's below-chance AUC
  (0.433) is a 32-window / 4-clip sample-size artefact, not an
  anti-predictive model; class 2's ~0.5 reflects the structural hardness
  of one-vs-rest scoring for the middle band of an ordinal scale. Both
  answers are in thesis §4.3.2.
- If asked "which checkpoint produced your test figures?": the frozen
  2026-08-02 checkpoint (test consumed exactly once); the shipped
  browser model is the later states-head retrain, deliberately never
  re-evaluated on Test — disclosed in thesis §4.3.2, not a discrepancy.
- The J1 CI job passes on GitHub by correctly *skipping* the actual
  parity assertion (the DAiSEE-derived fixture can't legally be committed
  there) — it has been exercised, and confirmed to catch TWO real defects
  (the numFaces regression and the Amendment 4 brow-formula divergence),
  locally. The export-parity gate DOES run fully in CI on every push.
  Say all of this plainly if asked "is this actually tested in CI."

## Go / no-go before tagging `v1.0`

The freeze rule is: *"Tag v1.0. No commits after this
except what the dry run forces."* Don't tag until:

- [ ] This checklist has actually been run live at least once, not just
      read.
- [ ] Anything it surfaced has been fixed or has an honest answer ready.
- [ ] Final README pass is current (done 2026-08-09 — re-check if the
      repo has changed since).
- [ ] You're ready to stop making non-essential commits — tagging is a
      commitment to "this is the version I defend," not a checkpoint to
      revisit casually.

Tag with:

```powershell
git tag -a v1.0 -m "FYP submission freeze"
git push origin v1.0
```

**Not done as part of this session** — deliberately left for you to run
once the live rehearsal above has actually happened.
