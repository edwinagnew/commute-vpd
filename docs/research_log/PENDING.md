# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-03 · decisions.md; reports/analyses/2026-11-03_analysis_qwen-line-story.md · section 3, overview Line 3, Q9, next steps, timeline · Remove card decision-e61-next (user chose option 1, stop and write up); overview and next steps say the Qwen line has stopped and link the whole-story write-up.
