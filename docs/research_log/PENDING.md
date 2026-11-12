# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-12 · reports/analyses/2026-11-12_analysis_third-wording-no-reading.md · P1.18 card (gate page addendum), P1.19 plan line, timeline · third reuse wording (testing only, not adopted) read in principle on every item: planted controls read as constructed and a new computation is readable positively; the memorised fact and every per-token new side get the grade the map class sets; fine-tunes may get no reading if no fixed variable carries the change. Plan for P1.19: reports/plans/2026-11-12_plan_soft-intervention-wording.md
