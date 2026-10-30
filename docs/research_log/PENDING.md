# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-31 reuse-lead · reports/plans/2026-10-31_plan_e55-stage-c-open-pair.md, decisions.md (update-size row) · E55 card (next step), timeline, decisions · plan for E55 C3: pruning revival and reverse fine-tuning ported to the 67M and read on E58's 12 open-pair states, beside r, the stand-in ratio and S, to ask whether the granularity limit is VPD's alone ($0, one cluster debug job; predictions drafted); new standing rule: update size within a recipe cell is a baseline column in every instrument comparison
