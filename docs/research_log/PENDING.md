# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · experiments/TRIED.md, decisions.md, reports/plans/2026-11-02_plan_e67-weight-side-read.md · changelog, re-gating's what's next · E67 coded, not run: all-position change in Γ's values and an overlap share against the learned-nothing fine-tune, one free job after a review; a decisions row records that no decision chose log-odds for the route split.
