# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · reports/analyses/2026-11-10_analysis_old-task-labels.md · new E142 card (cell P1.7, run 1 of 2; figure experiments/old_task_labels/figures/old_task_labels.png), claims table, definition sub-question status, timeline · provisional labels on the 17 old 67M tasks under the second wording: built-as-reuse and built-as-new tasks both read rewired (11 of 16 two-seed tasks), the real case label alone reads re-weighted, the two text ciphers, lowercase and the emoticon read no map found; G orders every matched reuse/new group by construction but under the filed cut-off; no rule, cut-off or wording changed
