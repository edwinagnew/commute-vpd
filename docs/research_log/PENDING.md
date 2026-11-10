# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 compare-lead · docs/research_log/parts/experiments/e133.html · Restructure on the user's request (parts already edited, apply nothing more): the baselines comparison is now one experiment card, E133, folding the cards of E135, E136, E140, E141, E143 and P2.11 (their ids kept as anchors); the figure is redrawn as fig/e133_final.png, coloured by call with the footer text cut; section methodtable's six paragraphs are one; TRIED.md notes the fold. Changelog row only.
