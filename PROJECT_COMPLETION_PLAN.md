# Project Completion Plan — from gap closure to submission

Companion to `BUILD_PLAN_1.md` and `GAP_CLOSURE_PLAN.md`. Written 2026-08-09. Covers everything between "gap closure merged" and "ready to submit and defend" — phased, with a checkpoint (gate) closing each phase, in the same spirit as BUILD_PLAN_1.md's J1–J4 joint gates: nothing here is "done" until its acceptance criterion is actually true, not just attempted.

**Status as of 2026-08-30: Phases 0, 1, 3 and 5.2 executed; Phase 2 (two
more benchmark machines), Phase 4 (student handoff) and Phase 5.3 (the
`v1.0` tag) outstanding. Phase 5.1's live rehearsal was completed
2026-08-29 — `docs/dry-run-checklist.md`. See `SUBMISSION_CHECKLIST.md`
for the consolidated remaining list.**

## Context

Per the status review: Track A and Track B's individual technical deliverables (A1–A9, B1–B9) are essentially complete, and once `GAP_CLOSURE_PLAN.md` lands, so are the joint gates (J1 parity, J2 e2e, J3 single-machine, A5 baselines) plus CI. What's left is everything BUILD_PLAN_1.md scoped for Days 16–20 plus the FYP's non-software half — the thesis itself, the multi-machine benchmark requirement (needs hardware this session doesn't have), and rehearsing the demo against real failure modes before a panel finds them.

**Three things in this plan need input only you can supply** — flagged inline, not guessed at: your university's thesis template/format, which two additional machines are actually available for J3, and your defense/submission date. Everything else is derived from the repo and BUILD_PLAN_1.md.

---

## Phase 0 — Gap closure (prerequisite)

This is `GAP_CLOSURE_PLAN.md` in full: J1 rebuild, A5 baselines, B7/J3 automation, J2 e2e fix, CI. Do this first — hardening a demo (Phase 1) on top of an unverified feature pipeline, or writing a results chapter (Phase 3) before `baselines.csv` exists, is wasted or reworked effort.

**Checkpoint 0:** CI green on GitHub Actions; `npm run test:parity` and the rewritten `e2e_app_test.py` both pass headless; `docs/results/baselines.csv` has real numbers for both splits; at least one fresh (real-model) entry in `docs/benchmarks/`.

---

## Phase 1 — Hardening & cross-browser

Goal: prove the demo survives contact with reality — a different machine, a different browser, an imperfect room — not just the dev machine it was built on. Maps to BUILD_PLAN_1.md's Day 16–17 buffer/cross-browser slots.

> **Status (2026-08-09): DONE**, all three sub-phases. Full findings in
> `docs/demo-failure-modes.md`; summary below each sub-phase. One real
> gap found and fixed along the way (README's torch install ordering);
> everything else tested clean.

### 1.1 Clean-machine install verification
Run the README's Track A and Track B setup instructions **verbatim**, on a genuinely clean checkout (fresh clone in a new folder at minimum; the student's machine or a VM is a better test since it's not primed with any of this session's cached state). Fix anything that doesn't work exactly as documented.

**Acceptance:** `npm install && npm run build && npm start` succeeds from a clean clone, matching the Definition of Done item directly.

> **Status: DONE.** Ran verbatim against a genuine fresh `git clone`
> (not this session's working copy — a real `origin` clone in an
> unrelated directory). Found the README's own documented torch
> workaround needs to be the primary instruction, not a fallback: a
> plain `pip install -r ml\requirements.txt` fails outright on a clean
> machine (`No matching distribution found for torch==2.13.0+cpu`)
> because that build only resolves via `--extra-index-url`. Fixed in
> README.md directly. Also hit (and worked around, not a doc bug) a
> Windows `MAX_PATH` failure installing torch from a deeply-nested clone
> path — noted in the README as a one-line caution. After the fix: clean
> install → build → start → real fake-webcam prediction, end to end.

### 1.2 Cross-browser check
Run the live app in Chrome (already the baseline throughout development), Firefox, and Edge at minimum; Safari too if a Mac is reachable — historically the roughest for WASM/getUserMedia/COOP-COEP support.

Watch for: whether COOP/COEP headers are honored (WASM threading silently falls back to single-thread if not — `PerfHUD.tsx`'s thread count will reveal this), and camera-permission UX differences across browsers.

**Acceptance:** app loads and produces predictions in every tested browser; record backend/thread-count per browser (feeds Phase 2's benchmark table).

> **Status: DONE** for Chrome/Edge/Firefox (Safari skipped — no Mac
> reachable, per this task's own conditional). All three report
> `wasm×20` — COOP/COEP cross-origin isolation, and therefore WASM
> multithreading, is honored in every browser tested, not just Chrome.
> Chrome and Edge got real face detection via a file-backed fake camera
> (Chromium-only Playwright capability) with identical results. Firefox
> has no equivalent in Playwright, so it ran Firefox's own synthetic
> fake-camera pattern instead of a real face — confirms the app loads,
> MediaPipe/WASM initializes, and the "no face" path renders correctly
> in Firefox, but does **not** independently confirm real-face detection
> there; flagged in `docs/demo-failure-modes.md` as needing a manual
> webcam test at some point before the defense.

### 1.3 Demo failure-mode hardening
Definition of Done, verbatim: "Demo survives: bad lighting, no face, glasses, two faces." Test each on purpose, don't wait for the panel to find them:

| Scenario | What to check |
|---|---|
| No face in frame | Features emit all-zero + `face_present=0` per CONTRACT §2.1's missing-face rule; UI shows a clear "no face" state, not a frozen stale prediction or a crash |
| Two faces in frame | Primary-face selection (`lib/primaryFace.ts`) picks sanely; overlay dims the non-primary face as designed |
| Bad lighting | MediaPipe either degrades gracefully or cleanly reports no-detection — never a silently garbage prediction |
| Glasses | Iris-landmark reliability specifically — glasses glare is a known MediaPipe weak point, and `gaze_x`/`gaze_y` depend entirely on iris landmarks |

**Acceptance:** none of the above crash or freeze the UI. Write actual observed behavior — including anything imperfect — to a new `docs/demo-failure-modes.md`. Knowing what breaks and saying so up front beats the panel discovering it live; this is literally BUILD_PLAN_1.md's own J4 principle, just applied before the deadline instead of during it.

> **Status: DONE.** All four scenarios plus a fifth (moderate vs. severe
> bad lighting, tested separately) ran clean: no crash, no freeze, no
> garbage prediction (every `engagement` vector summed to 1.0). Real
> DAiSEE clips throughout, except "two faces," where the natural
> candidate clip's background person never actually faces the camera —
> used a synthetic side-by-side composite of two real clips instead, so
> the multi-face + primary-selection + dimmed-overlay logic actually got
> exercised. One honest UX observation recorded (not a bug): there's no
> dedicated "no face currently visible" banner distinct from
> `WebcamFeed`'s camera-error banners — worth a callout during the Phase
> 5 rehearsal so it doesn't read as a freeze. Full detail:
> `docs/demo-failure-modes.md`.

---

## Phase 2 — Complete J3 (multi-machine benchmarks)

Uses the runbook `GAP_CLOSURE_PLAN.md` Part 3 produces (`docs/benchmarks/README.md` + `collect_benchmark.py --machine-label`).

1. **Needs your input:** which two additional machines are actually available — BUILD_PLAN_1.md's own suggestion is "both laptops plus one lab/library PC."
2. Run the benchmark script on each, drop the resulting JSON into `docs/benchmarks/`.
3. Build a small comparison table (mean/p50/p95/p99, FPS, backend/thread count across all 3 machines) — this is a genuine thesis result in its own right, not just a checkbox: BUILD_PLAN_1.md's own words are "'≥30 FPS' is meaningless without stating the hardware."

**Checkpoint 2:** three correctly-labeled, real-model benchmark JSONs in `docs/benchmarks/`.

> **Status (2026-08-09): Handed off, per your call.** This genuinely
> needs machine access this session doesn't have. Runbook is ready
> (`docs/benchmarks/README.md`, from `GAP_CLOSURE_PLAN.md`); exact
> command: `python ml\scripts\collect_benchmark.py --machine-label "..."`
> after `npm install && npm run build` in `web/` on each machine. Tracked
> as the one open item blocking full J3 completion in `docs/PROGRESS.md`
> and the Definition of Done table below.

---

## Phase 3 — Thesis writeup

> **Status (2026-08-09): First full draft complete.** Once
> `MSc Final Report Template Data Modelling.docx` (chapter structure) and
> `MScProject_Marking Scheme 2025_26.docx` (word count 10,000–15,000;
> Harvard referencing; four equally-weighted assessed sections) were
> supplied, §3.2's blockers were resolved except submission/defense
> dates. Full draft: `docs/thesis/FYP_Report.md` (source) and
> `docs/thesis/FYP_Report.docx` (generated, ready to open in Word) —
> 14,352 words main body / 16,400 total, mapped onto the template's exact
> chapter structure (Introduction → Literature Review → Methodology →
> Experimentation → Conclusion, each sub-numbered as the template
> specifies), 13 references in Harvard style — every one individually
> verified via web search before inclusion, none fabricated. Every
> figure/table/number in the draft is drawn from this repository's own
> produced artefacts (`docs/results/`, `docs/privacy.md`,
> `docs/demo-failure-modes.md`), fact-checked against the underlying CSV/
> JSON a second time during drafting (two numeric errors caught and fixed
> this way: a mixed-split percentage, and a size-reduction ratio).
> Remaining before submission: the bracketed placeholders (name, banner
> ID, supervisor, dates), inserting the actual figure images/table
> captions where marked, and a supervisor review pass.

The largest open item — `docs/PROGRESS.md`'s own open-items list names it first, and it's the one part of this whole plan with no code-level acceptance test. Nearly everything it needs already exists in `docs/results/`; this phase is assembly and writing, not new experiments.

### 3.1 Already available to draft from (all real, already produced)
- **Methodology / system design** — `CONTRACT.md`'s feature definitions, `ml/src/model.py`'s TCN architecture, `ml/src/train.py`'s training regime, `docs/architecture.md`'s stage-by-stage browser pipeline (once Phase 0 fixes its stale GPU-delegate line) are direct source material, not things to re-derive.
- **Results / evaluation** — `metrics_{validation,test}.csv`, confusion matrices, ROC curves, `class_dist.png`, `quantization.csv`, `baselines.csv` (Phase 0), the multi-machine benchmark table (Phase 2, still 1 of 3 machines — the draft states this honestly rather than assuming completion).
- **A known limitation to state plainly, not hide**: engagement class 0 has only 4 test clips (`docs/PROGRESS.md`), so its per-class metrics are statistically unmeasurable, and adjacent-class confusion dominates the errors. BUILD_PLAN_1.md's own risk register says a properly-analyzed negative/weak result passes review and a fabricated one fails — this is exactly that situation, and it is Section 4.3.3/5.2.2 of the draft.
- **Privacy** — `docs/privacy.md` is already a rigorous, evidence-backed section (real network capture, a caught telemetry call) — adapted directly into Section 4.4 rather than rewritten.

### 3.2 Needs your input (resolved except dates)
- ~~Your university's required thesis structure/template~~ — **supplied 2026-08-09**, followed exactly.
- ~~Page/word count targets~~ — **supplied 2026-08-09** (marking scheme: 10,000–15,000 words); draft is at 14,352/16,400.
- Submission deadline and defense date — still needed, only to pace the *remaining* steps (placeholder completion, supervisor review), not to write further content.

### 3.3 Checkpoints, mapped onto the actual template
- **3a** — Methodology / system design chapter drafted — **done** (`docs/thesis/FYP_Report.md` §3).
- **3b** — Results / evaluation chapter drafted, including the honest class-0 discussion and the A5 baseline comparison — **done** (§4, Experiments 2–3).
- **3c** — Privacy/ethics section drafted — **done** (§4.4, §3.3).
- **3d** — Full draft assembled — **done**; supervisor review pass — **not done**, needs the user/supervisor.

---

## Phase 4 — Student handoff

Definition of Done, verbatim: "The student can walk the full pipeline unaided and answer the questions in her prep plan." Because A5's baselines are being implemented on her behalf now (your call, not the original hand-off design), this phase matters *more*, not less — she should understand `baselines.py`, not just receive its CSV.

1. Walk her through the real pipeline end to end: extraction → features → windowing → baselines → TCN → export — using the actual code, not slides made about it.
2. Confirm she can independently answer the acceptance-criteria questions BUILD_PLAN_1.md embeds throughout (why macro-F1 instead of accuracy, why class weighting mattered, what the parity gate actually proves).

**Checkpoint 4:** she can run `python ml/src/baselines.py` and explain every line of the CSV it produces, unaided, on request.

> **Status (2026-08-09): Materials prepared, session itself still to
> happen.** `docs/student-handoff.md` is a stage-by-stage walkthrough
> guide (extraction → features → windowing → baselines → TCN → export,
> each with what to run and what to ask her) plus a question bank pulled
> directly from BUILD_PLAN_1.md's embedded rationale, and a deep-dive on
> `baselines.py` specifically sized to Checkpoint 4. This can't close the
> checkpoint itself — only the actual conversation with the student can —
> but the walkthrough and questions no longer need to be built from
> scratch in the room.

---

## Phase 5 — Freeze & dry run (J4)

Direct continuation of BUILD_PLAN_1.md's own Day 20 gate.

1. Full demo rehearsal that deliberately triggers the Phase 1.3 failure cases live — the first time you see them should not be in front of the panel.
2. Final README pass — every command in it re-verified after everything above has changed the repo.
3. Tag `v1.0`. No commits after this except what the dry run forces.

**Checkpoint 5:** tagged release, rehearsed demo including failure cases, full Definition of Done table below checked off.

> **Status (2026-08-09):**
> - **5.2 (README re-verification): DONE.** Every literal command in
>   `README.md` re-run against current repo state: venv + pip install
>   (including the fixed torch line), MediaPipe model downloads (both
>   copies), `npm install`, `npm run dev` (confirmed serving on
>   `localhost:3000` exactly as documented), the DAiSEE form URL (live),
>   and a full re-check of git history for stray video/large files (one
>   file over 1 MB in the whole history, a legitimate verification PNG —
>   citation and no-video-in-history claims both hold).
> - **5.1 (live rehearsal): script ready, not yet run.**
>   `docs/dry-run-checklist.md` is a live-rehearsal script built directly
>   from the automated `docs/demo-failure-modes.md` findings — what to
>   trigger, what to say, what's expected, plus a known-limitations
>   answer bank and a go/no-go list before tagging. It cannot substitute
>   for actually sitting in front of a real camera and running it.
> - **5.3 (tag `v1.0`): deliberately not done.** Gated behind the live
>   rehearsal above actually happening, per this phase's own ordering —
>   tagging first would invert the point of a dry run.

---

## Definition of Done — consolidated (BUILD_PLAN_1.md §10, cross-referenced)

| Item | Status now (2026-08-09, updated) | Closed by |
|---|---|---|
| Repo tagged `v1.0`; README works on a clean machine | README **done** (verified on a fresh clone, one real gap found + fixed, re-verified command-by-command); tag deliberately not done — gated behind the live rehearsal | Phase 5.1, 5.3 |
| J1 parity test in CI and passing | **Done**, with a caveat — see CONTRACT.md Amendment 2 / `GAP_CLOSURE_PLAN.md` | Phase 0 |
| `docs/results/` — confusion matrix, ROC, per-class metrics, baseline table, class dist | **Done** — `baselines.csv` added | Phase 0 |
| `docs/benchmarks/` — 3 machines, CPU/RAM/browser recorded | 1 of 3 — handed off, runbook ready | Phase 2 |
| `docs/results/quantization.csv` | **Done** | — |
| `docs/privacy.md` | **Done** | — |
| Live demo runs from clean `npm install && npm run build && npm start` | **Done** — verified 2026-08-09 on a genuine fresh clone | Phase 1.1 |
| Demo survives bad lighting, no face, glasses, two faces | **Done** — verified 2026-08-09, `docs/demo-failure-modes.md` | Phase 1.3 |
| Cross-browser (Chrome/Firefox/Edge) | **Done**, Firefox with a caveat (no real-face test possible in headless Firefox — needs one manual check) | Phase 1.2 |
| DAiSEE citations in README; no video files in git history | **Done** — both verified directly (citations present; git history checked clean of large/video blobs) | — |
| Student can walk the pipeline unaided | Materials ready (`docs/student-handoff.md`); the actual session is still to happen | Phase 4 |
| Live dry-run rehearsal | Script ready (`docs/dry-run-checklist.md`); not yet run for real | Phase 5.1 |
| *(not in original DoD, added by audit)* CI wired at all | **Done** | Phase 0 |
| *(tracked separately in `docs/PROGRESS.md`)* Thesis writeup | **First full draft done** (`docs/thesis/FYP_Report.md`/`.docx`, 14,352/16,400 words, Harvard refs) — placeholders + supervisor review remain | Phase 3 |

---

## Risk register additions (beyond BUILD_PLAN_1.md's own)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Thesis writeup is the single largest unscoped item left, with no code-level acceptance test | High | Start Phase 3 drafting in parallel with Phase 1/2, not sequentially after them |
| Only 1 of 3 required benchmark machines currently available | Medium | Phase 2 is explicitly blocked on your input — flag machine access now, not at J3 time |
| Commit velocity has already dropped once — last commit 2026-08-03, six days idle before this session | Medium | The Day-1-through-3 burst was unusually fast; don't assume the same pace holds for the writing-heavy phases (3, 4) that don't compress the same way code does |
