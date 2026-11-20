# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).

- 2026-11-19 · reports/decisions/2026-11-19_decision_p2-41-read-unit.md · pending decisions, cells table (add a P2.41 row to the part-2 table) · P2.41 run 4 (E222, reports/analyses/2026-11-19_analysis_e222-rl-readings.md): readings on the 27 size-matched RL/supervised/shuffled states make no call (only repeat RL, share 0.68-0.79, and the French anchor, 0.53-0.56, clear the floor); RL on lowercase reads negative because the rows scored the next word and RL wrote lowercase word salad; decision card: one more $0 read as a two-form contrast, or close
