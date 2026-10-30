# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 qwen-decomp-lead · reports/plans/2026-10-30_plan_q3-reading-the-update.md · E45 card (Run line: Amendment 2 filed as drafted and running in parallel; Due: E57 reads its states), timeline · E45 Amendment 2 filed as drafted (Fable's advice adopted; user asked that smaller choices go to Fable); E57 (Q3) planned to read every saved Qwen update on the cluster for $0
- 2026-10-30 qwen-decomp-lead · decisions.md, reports/plans/2026-10-30_plan_q3-reading-the-update.md · E45 card (Moved on: Job 2 dropped), timeline · E45 Job 2 (gated H100 reading, ~$15.50) dropped: placement and behaviour kept come from E57's cells for $0, behaviour kept scored as filed under reading A with reading B beside it; the least-squares pass row is not run because the pass test has no teeth (E44 Amendment 2)
