# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-22 compare-lead · reports/analyses/2026-11-21_analysis_procedural-rl-sft.md · P2.56 card (filled) · E261: under definition (b) RL and SFT read alike on sequence and brackets, no call on addition (RL 0.075-0.141 vs SFT ~0.05), all far above random (~0.01); E220 repeat RL-over-SFT gap does not carry over; the re-gating share read RL above SFT on sequence, disagreeing with R_b; trained-only Spearman with R_b: KL 0.73, effective rank -0.64, untrained share 0.64, re-gating share 0.56, prior -0.11; premise doubtful (R_b may follow output movement). Figure experiments/procedural_rl/figures/proxies_vs_rb.png.
