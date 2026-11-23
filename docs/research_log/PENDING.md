# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-23 P2.61 correction: the needed-set gaps are 0.0104 and 0.0130 (exact; v456 printed the rounded-mean differences 0.0105 and 0.0131). Page reports/analyses/2026-11-23_analysis_p2-61-uncapped-search.md.
- 2026-11-23 P1.55 (E274) filled: the squared-cosine share with one positive gain over F's own settings, over the needed set, is admissible on the 73 saved states (top only on the empty-K state) but tracks change size, not reuse: trained states 0.9935-0.9999, rank correlation with E_b over K -0.58, gain x10 lowest at 0.347; per variable its shortfall is the write share's times the change size. The write share stays the reading. Page reports/analyses/2026-11-23_analysis_e274-cos-admissible.md; admissibility page dated section. Log: P1.55 card, claims table (the squared-cosine reading), timeline 11-23.
