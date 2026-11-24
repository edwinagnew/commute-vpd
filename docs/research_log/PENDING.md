# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-24 · fleet/paper/cells.tsv (cell P2.76) · next · New cell P2.76 for regating-lead, 1 runs, due 2026-11-24: Figure 5 and the App. I difficulty figure standardised: every reading on the same tasks, gain read at matched states
- 2026-11-24 · fleet/paper/cells.tsv (cell P2.77) · next · New cell P2.77 for baseline-lead, 2 runs, due 2026-11-25: Figure 4 and App. I known_appendix standardised: 3 seeds a side on every known-answer pair, one task variant a side, one form of each reading
