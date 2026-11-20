# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).

- 2026-11-19 · paper/sec5_baselines_scorecard_caveats.tex (writeup repo) · cells table, timeline · P2.40 filled (zero-run): the paper's baselines, scorecard and caveats sections drafted, with redteam-lead's three corrections and compare-lead's four answers; no verdict, definition column waits on ask 50, P1.35, P1.36
- 2026-11-19 · reports/analyses/2026-11-19_analysis_e214-kstar-levels.md · E214 card, cells table, timeline · P1.35 run 1 analysed: the two hand-built reuse edits were not built (copy never reached its bar), so every filed rule is not read; the user's substitute reading with E195's reuse items puts the lowest reuse item above both built new edits under the read-position grade at all three coarse levels, beyond the new edits' own spread only at blocks, and below them under the every-position grade; no verdict on the definition
