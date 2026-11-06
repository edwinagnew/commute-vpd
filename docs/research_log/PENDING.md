# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/analyses/2026-11-06_analysis_d2-sft-vs-rl.md · D2 card or section line, timeline · qwen-rl-lead D2 synthesis of E75, E93, E104 on RL, shuffled-reward null and own-samples SFT: RL reads like its null at about the seed-noise rate (7 of 31 comparisons count, near the 22% chance rate) except an unfiled lead, no-think overlap 3.72-3.78 for RL against 2.77-3.15 for the null at both steps and seeds; own-samples differs from RL mostly by size and which matrices it sits in; one own-samples seed; teacher training (E111) and adjoint split (E93 A1) columns still to land; provisional, no independent check. Timeline: D2 synthesis written, no RL-specific reading beyond seed spread yet.
