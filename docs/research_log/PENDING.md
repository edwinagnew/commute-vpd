# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-12 · reports/analyses/2026-11-12_analysis_p3-7-library-cheap-readings.md · Part 3 / Qwen experiment card (E169), timeline · P3.7 read (three pods): the cheap readings on the Qwen pair library as side A minus side B, description only. Top-neuron overlap is higher on side A beyond its random band on copy minus cipher at both saves, facts at the matched save and both twins late; effective rank is lower on side A on both built pairs late. Wherever overlap passed its band side A was also the smaller weight change (cipher and invented facts 2.0 to 2.4 times their partners late), so no gap is told apart from update size. Steering has a band only on gold. No reuse/new label, no share, no verdict.
