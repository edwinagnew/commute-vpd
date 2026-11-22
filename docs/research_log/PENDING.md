# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-22 compare-lead: P2.59 filled (E265, ADL rerun with a lowercase control, $0): control passed at the edge; lens pair gaps follow definition (b)'s write-share gaps (same sign 8 of 9, rank correlation 0.83), re-gating share -0.28, crosscoder -0.92; provisional, not KL-controlled. Page reports/analyses/2026-11-22_analysis_e265-adl-column.md. Sections: next (P2.59 row: closed -> filled), timeline 11-22, claims (claim 2 baselines).
