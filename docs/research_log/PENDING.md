# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · compare-lead · reports/analyses/2026-11-21_analysis_e244-key-grid-baselines.md · E244 card (P2.52), E237 card's figure (four baseline panels added), claims table (baseline rows), timeline · neuron shares set copy apart (r_n 0.581-0.611 vs 0.654-0.723; top-neuron k64 0.024-0.028, all under 0.03), prior-matched cipher with cipher only at k64; crosscoder sets no task apart (seed spreads up to 0.43; copy 0.056/0.488/0.489, the expected 0.06 was E203's one seed), 13 of 15 fine-tunes below their random band; controls exact; figure experiments/key_grid/figures/columns.png
