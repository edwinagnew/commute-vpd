# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-27 · reports/analyses/2026-11-26_analysis_p2-87-subspace-share.md (E319, P2.87 filled) · main, section {{n:next}} P2.87 row; experiments page, re-gating share card · Random-subspace reference for the re-gating share on reuse tasks: updates trained only inside a fixed random subspace (rank-8 LoRA count) all reached 1 log-odds gain on the 8 reuse tasks. At that first state they read 0.39-0.47 below their own fine-tune on copying tasks and level or near it elsewhere; by step 300 every task was within about 0.10. The filed rule reads "rival a" by one task one batch spread over the bar; premise check doubtful (a trained reference may share the share because task and gain fix it). Figure experiments/subspace_share/figures/shares.png.
