# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · compare-lead · reports/analyses/2026-11-21_analysis_prior-mech-2x2.md · new E249/E250 card (P2.54, filled), claims table (prior-by-mechanism 2x2 row), timeline · Copy head (2,4) switched off labels copy and a low-prior copy (the copy task on its lowest-prior rows) reused and both ciphers new. On that 2x2 the re-gating share follows mechanism (effect 0.43 vs prior effect 0.07), and so do pooled steering, effective rank and top-64 neurons; only the prior follows prior. Caveat first: the low copy is a row selection, so every update reading lands near copy; head (1,1) carries as much as (2,4). Figure experiments/prior_mech_2x2/figures/prior_mech_2x2.png.
