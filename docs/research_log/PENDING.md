# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-20 · reports/analyses/2026-11-20_analysis_e226-kstar-vpd.md · experiments card E226, definition box "Readings so far", timeline · P1.37 filled: at VPD the known-answer rule reverses (x10 reuse R 0.9755 below new_thin new 0.9963, band 0.0010); cause is the squared-cosine grade; on the nine pairs R gaps under the band and rank with update size.
- 2026-11-20 · reports/analyses/2026-11-20_analysis_e232-size-matched-r.md · experiments card E232, timeline · P1.38 filled: at matched update size the block-level R gap keeps its sign on 5 of 9 pairs (filed at least 8); a size-only rule predicts every sign, so coarse-level R on these pairs is a size reading.
