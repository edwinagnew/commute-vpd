# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).


- 2026-11-16 definition-lead: in section next, P1.21 is closed (1 of 2 runs; cancelled by the user 2026-11-16 ~22:00 for the curves re-read), not open 0 of 2; fix its row status.
- 2026-11-17 qwen-lead: P3.11 filled (E186, 1 of 2 runs, $19.40 of $20): repaired Qwen share reference on p-015e0710 reads mixed under the filed rule (of 14 comparable built switch-offs, 5 above the random band, 7 inside, 2 below); bands widened to 0.046-0.154 as the three new random draws read lower (each draw shares one direction across row sets); fine-tunes 26 above, 10 inside; neither rival reaches 80%. Premise check: doubtful (the band moved rather than settled), for the 11-18 gate. Page: reports/analyses/2026-11-17_analysis_e186-share-reference-repair.md.
