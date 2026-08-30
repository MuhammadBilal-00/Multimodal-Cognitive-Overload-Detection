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
- [x] The figure/table lists were rebuilt to collect by paragraph style —
      as originally written they would have generated *empty* ("No table
      of figures entries found"), because Word's `\c` switch needs SEQ
      fields that manually-numbered captions do not have.
- [x] `code spans` nested inside **bold** no longer render with literal
      backticks (14 of them, including on the title page).
- [x] References and Appendices promoted to Heading 1 so the generated
      contents list does not file them under "5. Conclusion".
- [x] Harvard referencing repaired: 3 reference entries were never cited
      in the body and 2 in-text attributions had no entry (one was a
      leftover drafting note, "(research.google, cited in Section 2.2
      discussion)"). All 15 entries are now cited.
- [x] Abstract corrected: it still carried wording §3.4 explicitly
      retracts ("CI-integrated parity gate") and cited a superseded
      privacy trace with a self-contradictory claim (65-second session,
      "zero outbound requests", *plus* a telemetry call). It now reports
      the committed 75-second production trace: 39 requests, all
      same-origin.

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

## 1b. Things an audit found that only a human can settle (2026-08-30)

Read these before Turnitin. They are ordered by risk to the mark.

- [ ] **Authorship consistency — do this first, with the supervisor.** The
      report's cover names Ibtissam Merzouqi and positions its author as
      Track A. The documents the report sends the examiner to as Appendices
      A, B and G — `CONTRACT.md` and the GitHub repository
      (`MuhammadBilal-00/...`) — attribute Track A to "Bilal" and Track B to
      "Azeem". An examiner who follows the report's own appendix pointers
      lands on a repository naming two people, neither of them the author.
      Whatever the agreed position is (sole author with a named
      collaborator, or a declared group project), the cover, the
      Acknowledgements, the AI declaration, `CONTRACT.md`'s role
      attribution and sign-off table, and the Appendix G repository all
      have to tell the same story. **Nothing here was renamed** — that is
      a decision for you and Ilya Alexakhin, not a text edit.
- [ ] **Supervisor review pass.** Required by
      `PROJECT_COMPLETION_PLAN.md` Checkpoint 3d and listed as open in
      `docs/PROGRESS.md` and `README.md`, but it appeared on no checklist.
      Send `FYP_Report.docx` to Ilya Alexakhin **before** Turnitin — a
      supervisor read is also how the two items below get settled.
- [ ] **Declaration of originality / own-work statement.** The front
      matter has an Abstract, Acknowledgements and a Declaration of AI Use,
      but no "this work is my own / has not been submitted elsewhere"
      statement, which UK MSc templates almost universally require
      *alongside* the AI declaration. The exact wording has to come from
      the Greenwich template or handbook — it was not invented here.
      Add it as a `## Declaration` next to the AI declaration and
      regenerate with `python docs/thesis/md_to_docx.py`.
- [ ] **Ethics statement.** The repository's only ethics content is one
      paragraph in §3.3 about DAiSEE's redistribution licence. There is no
      ethics reference number and no statement of whether university
      research-ethics approval was sought or was not required — despite the
      project using a human-subjects video dataset and recording live
      webcam sessions during the rehearsal. Confirm with the supervisor:
      if a form was filed, append it as Appendix H and cite its number in
      §3.3; if none was required, say so explicitly in §3.3 rather than
      leaving silence. This was **not** asserted either way here, because
      it is not a fact recoverable from the repository.
- [ ] **Five of the thirteen tables have no caption**, so they will not
      appear in the generated List of Tables. Add
      `Table N.N — <description>` lines above them in
      `docs/thesis/FYP_Report.md` if you want full coverage.

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
