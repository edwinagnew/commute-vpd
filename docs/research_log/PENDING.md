# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-22 definition-lead · Section next, Part 1 table: add the missing rows P1.39 to P1.48 and P1.53 from `~/commute-agents/fleet/cell show <id>` (title, status, due, runs, page link, one-clause reading as each cell's fill/close text gives it; P1.53's page is reports/analyses/2026-11-22_analysis_reading-admissibility.md), and fix the count line to "49 of 49 cells done or closed" (rows P1.5, P1.6, P1.10, P1.11, P1.21 stay out of this table only if still absent from the table of work display; otherwise drop the parenthesis). No other section changes.
