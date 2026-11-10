# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 P1.2 filled: the saved E62 copy-against-lookup pair re-read under the second wording reads rewired on all four seeds (copy re-weighted share 0.70, 0.69; lookup 0.33, 0.29; cut-off 0.79); the share still orders the pair on every seed. Page reports/analyses/2026-11-10_analysis_e62-reread.md.
- 2026-11-10 · docs/research_log/parts/experiments/p211.html · card p211 rewritten for readability (user asked) · already applied in parts/: rebuild and republish only, plus a changelog row
