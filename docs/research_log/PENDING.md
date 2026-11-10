# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · reports/analyses/2026-11-10_analysis_p2-11-method-comparison.md + experiments/method_table/figures/final.png · new experiment card (P2.11 method comparison, the part-2 scorecard as one figure) + main-page part-2 summary + timeline · P2.11 filled: one figure, 13 pairs by 9 methods; the share separates all nine built pairs the same way, steering 5 same / 3 reversed, neurons at most 3, update size never, prior same way on all 8 read (one value a side); the card must explain each baseline plainly (what it computes, which rival it stands for, what it is blind to), as the user asked 2026-11-10
