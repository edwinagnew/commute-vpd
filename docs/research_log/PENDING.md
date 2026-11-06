# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 qwen-rl-lead · reports/plans/2026-11-06_plan_e120-pass-rate-pools.md (Amendments 1-2), reports/reviews/2026-11-06_review_e120-gold-error-guard.md · E120 card (running) + timeline · E120 sampling done ($15.72): pools 78 a side, P1 (≥ 100) misses and stands; N1 novelty passes (0.256 solved with thinking); a MATH grader bug (choice letters, units) cut never-solved problems; Amendment 2 regrades on CPU, projected n ≈ 115, then SFT.
