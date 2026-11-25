# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-24 · fleet/paper/cells.tsv (cell P2.80) · next · New cell P2.80 for baseline-lead, 4 runs, due 2026-11-26: Every model-diffing tool of tab:tools on the same fine-tunes: Fig 4's 16 tasks x 3 seeds plus matched randoms
- 2026-11-24 · reports/analyses/2026-11-24_analysis_p2-63-reuse-dial.md (cell P2.63 filled, E300) · results, part 2 · Reuse dial on four realistic tasks at one matched gain: no cheap reading beat KL in following the dial in one direction across tasks (filed rule: rival b); the definition reading puts every positive-gain SFT and RL fine-tune below the dial resolution, so it cannot order RL against SFT. Premise check doubtful. Whether the dial goes into the paper is the user's call (whip-lead asking).
