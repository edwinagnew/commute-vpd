# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-24 · reports/analyses/2026-11-24_analysis_p2-74-old-tasks.md · E292 experiment card; table of work (P2.74 filled); main page part 2 · P2.74 filled (descriptive): on 15 old 67M tasks no reading separates the emoticon from both other-target twins; the old tasks gain about ten times more than the realistic panel, and easier tasks did not make any reading separate them; premise check doubtful (twins may not be a valid negative).
