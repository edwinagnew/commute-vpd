# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · reports/plans/2026-11-04_plan_e84-rephrasings.md · plan; experiments card E84 · the user's choice after E81: TOFU's 800 rows trained in five wordings, tested on TOFU's held rephrasing; $0 on `debug`, about 1 h; coded and filed, not run.
- 2026-11-04 · experiments/TRIED.md · experiments card E84 · new "Coded, not run" row for E84, `experiments/qwen_tofu_forms/`.
