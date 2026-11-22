# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-22 compare-lead: P2.60 filled (E266 KL control, E267 cluster job 63454, independent recompute; all $0, descriptive): the lens's built-pair agreement with definition (b) reproduces exactly and survives output KL (rank correlation 0.70 vs 0.83), but the lens does not separate copy from prior-matched ciphers (p 0.44, 0.22) where all eight cheap readings did, and on procedural fine-tunes its partial rank correlation with definition (b) given KL is -0.31 (20 states). Not to be cited as tracking the definition in general. Page reports/analyses/2026-11-22_analysis_p2-60-adl-checks.md. Sections: next (P2.60 row: open -> filled), timeline 11-22, claims (claim 2 baselines: the lens caveat).
