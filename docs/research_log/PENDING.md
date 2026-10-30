# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 · reports/plans/2026-10-29_plan_performed-anchor.md (Amendment 1), experiments/TRIED.md (E53 Amendment 1 row), experiments/performed_anchor/results/predictions_amend1.json · E53 card (Running line) · E53 Amendment 1 code and predictions filed: narrow lowercase (50 title-case types, ranks 50-99) vs broad cipher (500 words, ranks 50-549); swap check gates the reading; outcomes kind / breadth / both / neither / unscored; not run yet.
