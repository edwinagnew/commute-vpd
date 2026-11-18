# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · reports/analyses/2026-11-18_analysis_p2-31-crosscoder.md · E203 card, P2.31 row, timeline · crosscoder baseline (Minder et al. 2025, one seed a side) does not separate the nine built pairs: reuse side higher on 5/9 as built, 4/9 size-matched (6/9, 6/9 with artefacts counted as fine-tune-only), filed 7/9 not met; planted control missed; random update reads most shared; figure experiments/crosscoder/figures/scorecard_crosscoder.png
