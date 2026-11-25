# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-25 · fleet/paper/cells.tsv (cell P2.82) · next · New cell P2.82 for regating-lead, 1 runs, due 2026-11-27: Random baseline for the log-odds re-gating share: random changes scaled until they move the trained answer by one log-odds, share read on them beside each fine-tune's (Fig known tasks)
- 2026-11-25 · reports/analyses/2026-11-25_analysis_p2-82-random-logodds-share.md (cell P2.82, E310) · filled · Random baseline for the log-odds re-gating share: at 12 log-spaced sizes up to 100x the fine-tune size, random changes never moved the trained answer by one log-odds on a reuse task; on four new-side cipher or random-target tasks they read shares 0.24 to 0.91, at or above the fine-tunes 0.21 to 0.27, at gain about 1 against 11 to 21. Redteamed; sent to writeup-lead for the 5.2 sentence.
