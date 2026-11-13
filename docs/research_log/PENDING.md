# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-13 · reports/decisions/2026-11-13_decision_fourth-wording-simplest-intervention.md · decisions, glossary · Replace the card decision-third-wording-gate with decision-fourth-wording (verbatim from ~/commute-data/scratch/definition-lead/card-fourth-wording.html; the third-wording recommendation is withdrawn because that wording is not total). Insert the glossary entry def-reuse-simplest verbatim from ~/commute-data/scratch/definition-lead/glossary-reuse-simplest.html right after def-reuse-variable. Update the "Waiting on you" line to name the new card (asked by the user).
