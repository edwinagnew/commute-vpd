# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 · reports/analyses/2026-11-05_analysis_e100-fed-or-written.md, notebooks/E100_fed_or_written.ipynb, experiments/TRIED.md (E100, E100 Amendment 1) · new E100 card (figure experiments/fed_or_written/figures/reverts_sum.png), claims table, mechanism-location sub-question status, timeline · mechanism-detection-lead: E100's filed per-part fed/written column read the poles as built but put every other task "mixed" at mlp3 (outcome partial; copy-24 and cipher-24 same side); post hoc, the sum of the 28 single-part reverts sorts copy maps (0.22-0.40) from key-table maps (0.62-0.98) and passed 10/10 on unread seeds 3-7 (Amendment 1, job 52883); independent check holds with caveat: seed stability only, every fed state a copy map, rank-1 (filed fed) reads written and was regrouped after looking, the spread written plant reads 0.64 / 0.72 / 0.86 by site so the bar is site-dependent. A pair-free beside column for definition-lead, not a label. Provisional, $0.
