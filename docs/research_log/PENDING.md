# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 · experiments/convergence_panel/results/predictions.json (E272 Amendments 1-8) · P2.66 card (ongoing) · E272 convergence panel, pre-read: stage 0 passed; the panel was widened to 18 tasks with joint blind ratings (Opus, GPT) filed before any added-task training; held-out Pile cannot be learned at matched KL under 3 recipes (best +0.0005 nats/token vs floor 0.014, probe 63813), so pure-continuation reuse tasks drop; every task's learned-on-held floor is now scored at its own read positions; the nine-task set keeps 5 tasks at matched KL and 7 at matched size (reuse tier: lower, plus british by size), so its tier test has power ~0.06-0.16 at tau 0.3 and the 18-task panel carries H1; mixtures are read at matched size. Nothing read or scored yet.
