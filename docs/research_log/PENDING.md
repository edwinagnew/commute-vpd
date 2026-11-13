# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-13 · reports/analyses/2026-11-13_analysis_e178-qwen-rl-readable-run3.md · E163 experiment card (P3.4, run 3 and the cell's close), P3.4 row in next.html, timeline · P3.4 (E178, run 3 of 3): the shuffled-reward null at 1e-6, meant to stop where its output change matched RL's step-75 state, collapsed on seed 0 at step 90 (answers ~4 tokens, training reward 0.002, 0.82x the masked error against a 2.09x target, no state saved) and on seed 70 was stopped at step 20 by the filed reward-change check (reward 0.82 to 0.51); the filed verdict is "no call". The cell closes with RL's held-out gain at 2x (0.88 against W0's 0.78) never compared with a working null, as the null collapsed at both rates; every read update's outside share is as random. Cause of the collapse not checked. Figure experiments/qwen_rl_readable/figures/run3.png.
