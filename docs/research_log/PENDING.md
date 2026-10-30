# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-30 · `reports/analyses/2026-10-30_analysis_e58-open-pair.md` § Amendment 1 (+ `notebooks/E58_open_pair.ipynb`, `experiments/TRIED.md` E58 Amendment 1 row) · E58 card, re-gating claims table, step-2 sub-question status, pending-decision card for the held-out label, timeline 2026-10-30 · Job 46332 ($0): E58's elicit update is a lookup at unseen keys, not a sharpening (A1 pass); the held-out label's positive control FAILED as filed (copy map "reuses" on 1 of 3 seeds, x 0.42-0.55; rank-1 map not testable, 1 novel key), verdict "none", so the label is not used against r or the ratio; r and the ratio stay level (0.76-0.85, 0.93-1.11) while carry runs copy > rank-1 > elicit, separating only the cipher; lead consulting Fable.
