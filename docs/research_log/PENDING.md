# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/analyses/2026-11-06_analysis_e114-locator-case-label.md, notebooks/E114_case_locator.ipynb, experiments/TRIED.md (E114 row) · new E114 card; claims table; reuse sub-question status; timeline 2026-11-06 · E114 ran E97's locator and E100's sum of single reverts on E112's four case-label states ($0, job 54238): no part located on either unit, both arms need the same early parts, agreeing with the definition's "not through C"; but the sum of single reverts puts the real fine-tune with E100's copy rows (0.37/0.48) and the twin with its key tables (0.72/0.68), a disagreement for baseline-lead to weigh, untested against update size.
