# Submission checklist — viva 3 September 2026

Everything code/evidence/thesis-side is **done, committed, and pushed**
(see `docs/PROGRESS.md` for the full record). The thesis front matter was
finished on 2026-08-30 — the `.docx` now opens submission-clean, with no
placeholders and no manual Word surgery required.

What remains is below. Tick them off in order.

## 1. Thesis final touches (Word, ~5 min) — MOSTLY DONE

Done on 2026-08-30 (commit `docs(thesis): finish the front matter`):

- [x] **Acknowledgements** written.
- [x] **Table of Contents**, **List of Figures**, **List of Tables** are
      real Word field codes, and `w:updateFields` is set — Word builds all
      three automatically the moment the document opens. (If Word's
      security prompt suppresses that, press `Ctrl+A` then `F9`.)
- [x] Page numbers added — centred "Page N of M" footer.
- [x] All bracketed *[Word: …]* / *[fill in]* / *[REQUIRED …]* instruction
      lines removed from the document body.
- [x] Submission date on the cover verified: **3 September 2026**.

Still needs a human:

- [ ] Open `docs/thesis/FYP_Report.docx` in Word once and **accept the
      field update prompt**, then eyeball that the three lists populated
      and the page breaks fall sensibly. Save.
- [ ] **AI declaration** — check the "Declaration of AI Use" section's
      *wording and required placement* against Greenwich's generative-AI
      policy (programme handbook / Moodle, academic-integrity section);
      confirm with supervisor Ilya Alexakhin if ambiguous. The content is
      accurate and complete — only the university's required format needs
      confirming. (This was previously a bracketed note inside the report
      itself; it has been moved here so nothing bracketed is submitted.)
- [ ] Optional: personalise the Acknowledgements — it is a complete,
      accurate paragraph, but it is the one part of a dissertation that
      should sound like you.

## 2. Turnitin (do this EARLY)

- [ ] Submit the finished document to Turnitin via the module's submission
      point; leave enough days to react to the similarity report before the
      deadline.

## 3. Benchmark machines (≈20 min per machine)

The project's own Definition of Done wants 3 machines.

- [x] Machine 1 — `docs/benchmarks/benchmark-dev-i7-13700H-16GB.json`
      (Windows, Intel i7-13700H, 16 GB).
- [ ] Machine 2 — an Apple M4 / arm64 / 16 GB run was attempted on
      2026-08-30 but **not committed**: the only numbers obtainable were
      collected under heavy concurrent CPU load and varied by 2.5x across
      three consecutive runs (p50 0.67 / 0.60 / 0.26 ms), which is not a
      defensible hardware data point. Re-run it on a quiet machine with
      `python ml/scripts/collect_benchmark.py` (the collector was fixed on
      2026-08-30 — it previously could not find its own button).
- [ ] Machine 3 (friend's laptop, lab PC): `cd web && npm install &&
      npm run build`, then from the repo root `python
      ml/scripts/collect_benchmark.py --machine-label "<cpu>-<ram>"`
      (needs Python + `pip install -r ml/requirements.txt` with the
      `--extra-index-url` from README, + `playwright install chromium`).
      Commit the JSON that lands in `docs/benchmarks/`.
      If a third machine is not realistic, **do nothing further** — §4.5.3
      and §5.3 state the count honestly and stay accurate either way.

## 4. Azeem (one conversation)

- [ ] Sign-off boxes in `CONTRACT.md`: **Amendment 3** (states channel
      order) and **Amendment 4** (brow formula fix) — Bilal's boxes are
      ticked, Azeem's are open.
- [ ] The **student handoff session** (`docs/student-handoff.md` is the
      walkthrough script + question bank; checkpoint = she can run
      `python ml/src/baselines.py` and explain every column unaided).

## 5. Viva prep (before 3 September)

- [ ] Read `docs/viva-pack.md` twice; say the §1 three-sentence story and
      the Q2 answer ("why ship the TCN when gradient boosting beats it")
      out loud until fluent.
- [x] Live rehearsal DONE (2026-08-29, all failure modes passed —
      recorded in `docs/dry-run-checklist.md`). Optionally re-run once on
      the actual presentation machine/room if different (checklist Part 3).
- [ ] Demo start command: `cd web && npm run build && npm start`
      (port 3000; use `-p 3010` if 3000 is busy).
- [x] Live Stats readings recorded in `docs/dry-run-checklist.md`:
      render 36 fps, sampling 10 Hz, p50 0.78 ms, wasm×20.

## 6. Freeze (last step, after 1–4)

- [ ] Tag the release: `git tag v1.0 && git push origin v1.0`. Per
      `PROJECT_COMPLETION_PLAN.md` Phase 5: no commits after the tag except
      what a re-run of the dry-run forces.

---

*Not required / consciously skipped*: the Firefox real-webcam check (demo is
Chrome; the thesis states the gap honestly in §5.2.2) · committing the
university's marking-scheme/template `.docx` files (internal university
documents, deliberately untracked).
