# Submission checklist — viva 3 September 2026

Everything code/evidence/thesis-side is **done, committed, and pushed** as of
2026-08-29 (see `docs/PROGRESS.md` for the full record). What remains is
below — every item needs a human, none needs engineering. Tick them off in
order.

## 1. Thesis final touches (Word, ~30 min)

- [ ] Open `docs/thesis/FYP_Report.docx` in Word.
- [ ] **Table of Contents**: References → Table of Contents (headings are
      proper Word styles — one click).
- [ ] **List of Figures / List of Tables**: References → Insert Table of
      Figures, once with caption label "Figure", once with "Table" (all
      captions are real Word Caption paragraphs — generates automatically).
- [ ] Write the **Acknowledgements** paragraph.
- [ ] Delete the two bracketed *[Word: …]* instruction lines after using them.
- [ ] Verify the **submission date** on the cover (currently 3 September
      2026 — change if the document upload deadline differs from the viva).
- [ ] **AI declaration**: check the "Declaration of AI Use" section's wording
      against Greenwich's generative-AI policy (programme handbook / Moodle,
      academic-integrity section); confirm with supervisor Ilya Alexakhin if
      ambiguous. The content is accurate — only the required format/placement
      needs verifying.

## 2. Turnitin (do this EARLY)

- [ ] Submit the finished document to Turnitin via the module's submission
      point; leave enough days to react to the similarity report before the
      deadline.

## 3. Benchmark machines (≈20 min per machine)

The project's own Definition of Done wants 3 machines; 1 is recorded
(`docs/benchmarks/benchmark-dev-i7-13700H-16GB.json`). On each additional
machine (friend's laptop, lab PC):

- [ ] Machine 2: `cd web && npm install && npm run build`, then from the
      repo root: `python ml\scripts\collect_benchmark.py --machine-label
      "<cpu>-<ram>"` (needs Python + `pip install -r ml\requirements.txt`
      with the `--extra-index-url` from README, + `playwright install
      chromium`). Commit the JSON that lands in `docs/benchmarks/`.
- [ ] Machine 3: same. (If only one extra machine is realistic, run that one
      and keep the thesis's honest "n of 3" wording in §4.5.3 accurate.)

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
- [ ] The live rehearsal is DONE (2026-08-29, all failure modes passed —
      recorded in `docs/dry-run-checklist.md`). Optionally re-run it once on
      the actual presentation machine/room if different (checklist Part 3).
- [ ] Note for the demo: start the server with
      `cd web && npm run build && npm start` (port 3000; use `-p 3010` if
      3000 is busy).
- [ ] Photo/note of the live Stats readings is already recorded in
      `docs/dry-run-checklist.md`: render 36 fps, sampling 10 Hz, p50
      0.78 ms, wasm×20.

## 6. Freeze (last step, after 1–4)

- [ ] Tag the release: `git tag v1.0 && git push origin v1.0`. Per
      `PROJECT_COMPLETION_PLAN.md` Phase 5: no commits after the tag except
      what a re-run of the dry-run forces.

---

*Not required / consciously skipped*: the Firefox real-webcam check (demo is
Chrome; the thesis states the gap honestly in §5.2.2) · committing the
university's marking-scheme/template `.docx` files (internal university
documents, deliberately untracked).
