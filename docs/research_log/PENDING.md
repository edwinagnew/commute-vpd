# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-02 · experiments/teacher_training/results/teacher_training_amend8b_*.json and commit 11b5ca7 (E34 Amendment 8b, 8c: null student at a second batch order, cross-teacher cosines; job 48613) · E34 card, Q1 · found in the repository during the E62 flush with no queue line and no analysis report; the owner should queue it with its report.

