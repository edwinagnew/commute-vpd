# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/plans/2026-11-06_plan_e103-known-answer-pair.md (section "Amendment 1"), experiments/TRIED.md (E103, Amendment 1 row) · experiments page (E103 card: add "Amendment 1 coded, not run": pair 0 answered in a sentence so both arms share the format; the gain read without the end-of-turn token, bar half the fictional matched state's) · E103 Amendment 1 coded: the known-answer fine-tune is rebuilt with matched answer format, and its gain control now excludes the end token.
