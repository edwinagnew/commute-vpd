# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-09 · experiments/prior_vs_share/results/stage2.json · method table section, what is next, timeline · Cell P2.3 filled (E133): over 24 saved maps the original model's prior orders them much as the re-gating share does (Spearman 0.85, interval 0.67 to 0.93, reading agree); copy maps sit above the cipher and lookup at a close prior; labels as built, no verdict
- 2026-11-09 · reports/plans/2026-11-09_plan_p2-4-card.md · what is next · Cell P2.4 blocked: its card needs 3 seeds per side and 2 are saved; asked the user
