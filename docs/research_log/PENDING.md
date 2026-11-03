# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 · decisions.md · decisions · remove card decision-counting-large-direction: ask 17 taken by default at 18:08, E70 Amendment 2 runs (largest direction with no covariance with the count, LOW first, then P and N).
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e70-existing-new-counting.md · experiments, timeline · rewrite the E70 card with Amendment 2 (ran, job 50643, the last amendment): filed branch "count-specific at this size", label specific at blocks 1, 3, 4, 5, "cannot serve" stands; at blocks 3-5 the covariance direction costs LOW 8-12 log-odds and P more than N while the largest direction with no covariance costs nothing, unmatched on size; block 1 is not the count, block 6 carries the classes, N's loss unexplained, no independent check; figures `e70_a2_tasks.png`, `e70_a2_low.png`; timeline line for E70's end, no further amendment proposed.
