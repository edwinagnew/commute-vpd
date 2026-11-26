# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-25 · fleet/paper/cells.tsv (cell P1.64) · next · New cell P1.64 for litreview-lead, 0 runs, due 2026-11-26: Lit review: EH and alignment work (personas, alignment pretraining, safety cases, counter-evidence)
- 2026-11-25 · fleet/paper/cells.tsv (cell P2.83) · next · New cell P2.83 for baseline-lead, 3 runs, due 2026-11-26: Crosscoder fixed on Fig 4's 48 states: planted control passes under the authors' rule
- 2026-11-25 · fleet/paper/cells.tsv (cell P2.84) · next · New cell P2.84 for baseline-lead, 2 runs, due 2026-11-26: Reverse fine-tuning (Jain et al. 2024) on Fig 4's 48 states
