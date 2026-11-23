# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-22 · reports/decisions/2026-11-22_decision_e272-erank-tolerance.md · section 3 pending decisions (Blocking card) · P2.66 / E272 realistic convergence panel: stage 0 stopped on an effective-rank re-read bar (absolute 1e-6) under float noise (6e-7 relative); ask 27: make it relative 1e-4 and rerun stage 0 (recommended); share and steering re-reads passed
- 2026-11-22 · parts/sections/decisions.html · decisions · remove the E272 effective-rank card (ask 27 answered: whip-lead 21:55 (user away): erank bar relative 1e-4 (abs 0.0598), stage 0 rerun; E272 Amendment 1)
