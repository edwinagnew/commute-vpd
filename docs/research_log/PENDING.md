# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · reports/analyses/2026-11-18_analysis_realistic-tasks-run3.md · P2.38 row, E211 card · P2.38 filled (3 of 3 runs): on the real-text pairs the share separated only lowercase vs reversed (label no call, French unreadable) and no cheaper reading did better; the crosscoder failed its do-nothing control, real sides not run. Premise doubt (fleet/premise): with so few calls on real text, the L2.3 paragraph on what the share adds rests on built pairs.
- 2026-11-18 · parts/sections/decisions.html · decisions · update the card "Question for you · writeup-lead" (proposal 35): the four items are done and pushed to the paper repo (eaeaaf4: 11 bib entries and EH candidates, Table 1 of the 13 pairs, Fig. 1 schematic and Fig. 5, a clean pdflatex+bibtex build on the cluster worker via cpu-run, no Slurm job needed); approving now only records the cell (urgency No rush; "Nothing runs until this is decided" becomes "Nothing left to run"). Line for the user: ~/commute-agents/fleet/cell approve 35
