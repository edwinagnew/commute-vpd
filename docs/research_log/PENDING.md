# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-18 · reports/analyses/2026-11-18_analysis_realistic-tasks-run2.md, experiments/realistic_tasks/figures/run2_gaps.png · experiment card E211 (new, under P2.38 beside E209), claims (Part 3 row on the share separating the nine built pairs: add that on real-text pairs it carried to one of three), timeline 2026-11-18 · compare-lead: E211, every method on the three real-text pairs at two matched sizes, 3 seeds, controls passed. The share reads apart, reuse side higher, only on lowercase vs reversed order (gap 0.29 and 0.21); no call on topic vs coin label at the larger size; French vs Yoruba unreadable for every share-like method because French gains under 1 log-odds. Steering splits two pairs in opposite directions; erank reverses its filed direction on the label pair. Card reads rival b at both sizes; descriptive, no reused/new verdict; run 3 (crosscoder, E212) in preparation.
