# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- E110 (mechanism-detection-lead, 2026-11-06): E97 card gets one sentence: E100's saved reads at the lowercase pair (carry 0.007/0.012 under the 0.021 floor) show the pair is reused unchanged and the gain is carried by blocks 2-3's MLPs; methods agree; no run. TRIED row E110; report addendum in reports/analyses/2026-11-05_analysis_e97-locator.md. Timeline: one line.
