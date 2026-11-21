# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · reports/analyses/2026-11-21_analysis_p1-42-r-reanalysis.md · experiments card E238, definition box "Readings so far", timeline · P1.42 ($0 reanalysis): weighting R by behaviour does not fix the built edits (x10 0.9378 below new_thin 0.9974); R with update size removed orders all six (margin 0.57, band 0.25, any size-law slope 0.89 to 2.9) and puts reuse higher on 7 of 9 pairs, no longer following size; behaviour change from M also orders the six, so the built items cannot yet separate the two; a lead, not a pass.
