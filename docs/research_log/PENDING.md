# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/analyses/2026-11-06_analysis_e121-copy-gap-ladder.md (notebook notebooks/E121_copy_gap_ladder.ipynb, TRIED E121 row) · new card E121; teacher-training card E34 (Amendment 10 gap explained); claims table; timeline · E121 ran (job 54638, $0): teacher training's distance from δW (cosine 0.40 vs rerun 0.83) comes mostly from training U, V only (drop 0.28) and partly from the lower lr (0.19), none from the KL target (plain student 0.87); its change needs its remainder (P 0.28 without it) and its trigger logit change is further than a rerun's (0.64 vs 0.86); from the least-squares split it stays close (0.81) with gap 0.0035 vs 0.029. Claims on teacher-trained decompositions: keeps the target's probability, not the update or the function.
