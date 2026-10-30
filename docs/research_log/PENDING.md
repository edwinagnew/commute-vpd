# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-30 qwen-decomp-lead: E57 card (Blocking for Stage D): row 1 has no teeth at scale, both random controls read re-gating share ~0.99; options A-D and Fable advice in reports/decisions/2026-10-30_decision_e57-no-teeth.md; E57 experiment card: B2 and C0 done, C running.
