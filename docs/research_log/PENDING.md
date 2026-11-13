# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-13 · reports/analyses/2026-11-13_analysis_p3-8-library-share.md · P3.8 cell row (filled), timeline · P3.8 (E174): the re-gating share on p-9707d6ff reads 0.990-0.9998 on every side of the three library pairs (3 seeds, late and matched saves) and 0.992-1.009 on random changes of matched output change; largest side A minus side B gap 0.0032, all six cells inside their random band: the share is saturated on this decomposition and separates no pair. Description only, no labels. Premise check: doubtful (real updates read like random changes here).
