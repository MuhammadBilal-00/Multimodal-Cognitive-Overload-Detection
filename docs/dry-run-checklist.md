# Freeze & dry-run checklist (J4 / PROJECT_COMPLETION_PLAN.md Phase 5)

BUILD_PLAN_1.md's own J4 principle: *"Know what breaks before the panel
finds it."* This is a script for the **live** rehearsal — sitting in
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
| Step out of frame | "Watch the feature panel zero out, not freeze" | All 13 features → 0.0000, `face_present=0`, dashboard stays live, no crash |
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

Say these before you're asked, per BUILD_PLAN_1.md's own risk-register
philosophy (a properly-analyzed weak result passes review; a hidden one
doesn't):

- Engagement class 0 has only 4 clips in the entire test split — its
  per-class metrics are statistically close to meaningless alone; the
  3-class-merged metric exists specifically because of this
  (`docs/PROGRESS.md`).
- Only 1 of the 3 required benchmark machines has been run
  (`docs/benchmarks/README.md` has the runbook for the other two — if
  they're done by defense time, update this line).
- Macro-F1 numbers (0.2475 fp32 / 0.2460 int8 test) are modest in
  absolute terms; the comparison that matters is against the
  majority-class floor (0.1655) and the classical baselines (`baselines.csv`)
  — the model demonstrably uses temporal structure they can't.
- The J1 CI job passes on GitHub by correctly *skipping* the actual
  parity assertion (the DAiSEE-derived fixture can't legally be committed
  there) — it has been exercised, and confirmed to catch a real defect,
  locally. Say this plainly if asked "is this actually tested in CI."

## Go / no-go before tagging `v1.0`

Per PROJECT_COMPLETION_PLAN.md Phase 5: *"Tag v1.0. No commits after this
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
