# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 · decisions.md · sections/decisions.html · Card `decision-counting-one-direction` removed: ask 15 taken by default at 15:28 after two hours without an answer (the covariance-chosen direction, LOW first).
- 2026-11-03 · reports/plans/2026-11-03_plan_e70-existing-new-counting.md (Amendment 1) · experiments/e70.html · E70 card's running line: Amendment 1 coded, not run; removes the direction the count moves the residual along, on the pretrained model first, $0, about 17 minutes.
