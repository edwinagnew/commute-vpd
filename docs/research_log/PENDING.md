# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-11 · reports/analyses/2026-11-11_analysis_h-diagnosis.md · built-control card (E154/E156), next.html P1.13 row, timeline · P1.13 filled: no call; the reuse search reads only 0.84 on an edit built to be exactly reachable, wider maps do not raise the no-teaching control by the 0.2 bar, two sites in order put it just above 0.5; the shortfall is in the scale of H, not the fine-tunes. No wording change.
