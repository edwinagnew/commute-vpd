# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · definition-lead · glossary boxes def-existing-new, def-reuse-variable, def-reuse-simplest; the Part 2 cards in sections/overview.html and sections/next.html; decisions.md · The user revised the working definition in chat on 2026-11-18 (reuse = mean rule retention of the parts F needs for a behaviour, at a declared level; the held model is one cheap reading, not the definition); the overview box existing-new already says so (changelog 371). These places still call the held-model wording the working definition: give them the new status once definition-lead records the decision row in decisions.md.
