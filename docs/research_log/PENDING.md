# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 · experiments/TRIED.md (E53 row), experiments/performed_anchor/results/predictions_a0.json · E53 card (ongoing), timeline · E53 stage A0 coded and filed (W0 only: lowercase ≥ +1, cased ≤ −1, cipher ≤ −3 nats; stop if control misses, lowercase ≤ 0 or cipher > −1); not run yet

