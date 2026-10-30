# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 baseline-lead · reports/plans/2026-10-29_plan_e54-keydependent-elicit.md (Amendment 1), experiments/TRIED.md · E54 card (What next / Not run) · E54 Amendment 1 coded, not run: S read in log-odds (the route split's unit) on the six saved states, so the ratio table's "S estimated in log-odds" row becomes a measurement; predicted S_lo elicit ~0.25 vs matched cipher 0.22/0.18 (no separation), r - S_lo ~0.5 on elicit; MATS debug, $0.
