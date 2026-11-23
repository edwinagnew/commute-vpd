# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 regating-lead · reports/analyses/2026-11-23_analysis_convergence-panel.md · E272 card · redteam E_b verdict folded in: across the twelve tasks base loss does not account for the write-share reversal, within continued text all-position base loss does (partial -0.17 per byte, +0.14 per token); reading depends on the state (needed set 200 to 37,888 between seeds; consensus tau -0.69 all states, -0.59 without the four under 1,000); random update leans the same way weakly (tau -0.29), so most of the reversal comes from training; base loss's seven-task 0.873 and the 0.39 per-token partial are not evidence
