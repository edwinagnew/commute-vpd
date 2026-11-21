# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · reports/analyses/2026-11-21_analysis_p1-43-size-normalised-r.md · experiments card E239 (new, or folded into E226's card), definition box "Readings so far", timeline · P1.43 (E239, $0): size-normalised R, N = 1 - mean over changed circuit parts of (1 - s)/(relative weight change): filed rule holds, x10 0.233 above new_few -2.04 and new_thin -3.21 (gap 2.27, band 0.081), all four reuse edits above both new; but N is unbounded (t > 1 on most parts, because the grade weights inputs equally and the ratio does not) and on the nine pairs it follows update size the other way (reuse higher 5 of 9, gap vs size gap Spearman -0.92, states +0.62), as the crude size division did; the input-ratio form fails (x10 0.816 below new_thin 0.896)
