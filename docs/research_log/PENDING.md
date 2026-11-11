# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-11 · reports/analyses/2026-11-11_analysis_e152-qwen-instrument-check.md · E152 card (new, Qwen instrument check), next.html P3.3 row, Qwen part status, timeline · P3.3 filled: no row of the brief's outcome table called. The split returns a change planted on often-on subcomponents (0.89 on the set, chance 0.06) but only partly one on never-on subcomponents (recovery 0.29); the already-on gap 0.479 sits under the filed 0.5, so the split reads no call. The re-gating share separates states at every size (24 of 36 pairs), so row 4 (drop it) is ruled out, but the planted kinds swap order with size. No verdict on reuse.
