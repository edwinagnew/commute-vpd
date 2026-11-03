# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-03 · `reports/analyses/2026-11-03_analysis_method-table-section.md` · section `methodtable`, changelog · the 67M method table condensed to one write-up section for `writeup-lead`, one row per kind of truth; no new run, number or claim.
- 2026-11-03 E72 job 50428 stopped on room on u1 (filed expectation missed: ceiling gain 0.20 vs do-nothing 0.24); Amendment 1 reads u2, u3 and the clean reads; update card decision-d1-test's Running line and evidence.
