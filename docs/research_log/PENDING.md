# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 (qwen-rl-lead): remove pending cards decision-anchor2 and decision-e69-third-seed (user decided: TOFU; the two seeds as they stand; `decisions.md`). New card E81 stub: anchor 2, TOFU, plan `reports/plans/2026-11-04_plan_e81-tofu.md`; timeline line when stage 0 reads.
