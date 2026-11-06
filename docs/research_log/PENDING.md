# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 regating-lead · reports/analyses/2026-11-06_analysis_e119-teacher-route-split-triggers.md · experiments page: new E119 card (done, stopped by the user's reproduction rule; figure experiments/teacher_route_split/figures/route_labels.png); TRIED row; timeline line · E119: the regeneration of E34's teacher-trained copies did not reproduce within 0.05 (emoticon Adam Pile-text copy P 0.744 vs 0.902), so the run stopped; provisional readings of the other 11 on E12's lines: 8 neither like their fine-tunes, 3 Pile-text copies re-switching (r0 both, r1 SGD); no amendment.
