# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 definition-lead · experiments card E116 (new, parts/experiments/e116.html, figures fig/e116_possession.png and fig/e116_carrying.png) + timeline 11-06 one sentence · reports/analyses/2026-11-06_analysis_e116-definition-table.md: the old 67M tasks read under the frozen definition (rows 81, 87, 164): every row with a readout "unreadable" at possession, lowercase/emoticon/rand1 "does not apply", no earlier claim agrees or disagrees; carrying alone reads (copy maps pass, cipher and guess maps fail); the unreadable rows state the shape step 2 needs. Timeline: "E116: the frozen definition reads no old 67M task; it can read only a capability not fixed by one token type, with few values and answers for unseen values."
