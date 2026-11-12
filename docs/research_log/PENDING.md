# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-12 · reports/analyses/2026-11-12_analysis_memorised-fact.md · Part 1 / working-definition experiment card (E168), timeline · P1.16 filled: memorised fact vs last-number-odd twin (cluster, $0). The memorised-fact fine-tunes read well under their twin on the re-weighting share (0.10, 0.12 vs 0.39, 0.40), as in all five old matched pairs, but the planted fact and planted odd edits sit within 0.17 of each other and the exact planted odd edit reads only 0.63; the adopted wording labels the fact "no map found" (as the twin), the tentative "new = not re-weighted" reading labels every item new.
