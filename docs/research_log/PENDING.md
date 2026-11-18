# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-17 · reports/analyses/2026-11-17_analysis_share-size-matched.md · E204/E207 card (P2.34, closed), the P2.34 row, the re-gating share's sub-question status, timeline 11-17 · Both runs stopped at filed checks before any fine-tune, no share read: Always on scaled up stops re-weighting (one token takes most windows) and never reaches Table's gain; Table scaled down to Always on's gain stops doing its lookup. No call on kind vs size; P2.33's reading stands (the one split coincides with the one large change). $0.32; cell closed, no rescue run.