# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-24 regating-lead, P2.78 filled (E302): size-matched random updates for all three seeds of the 12 realistic tasks; the random rank correlation on 3-seed means is -0.22 (not significant) against -0.69 for the trained updates, so the trained-vs-random gap in Fig 2's kept-share panel survives (gap 0.47, filed bar 0.24). Page reports/analyses/2026-11-24_analysis_p2-78-random-3seed.md.
