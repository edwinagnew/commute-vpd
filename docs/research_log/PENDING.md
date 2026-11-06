# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/analyses/2026-11-06_analysis_e117-e118-neurons-beside-vpd.md (independent check fixes) · E117 and E118 cards, if already flushed · K_VPD = 0 on the next-token states is now stated as measured (no window-mean change above 0.3, single changes reach 1; spread thin or averaged away is open, k0_read.py written, not run); the known-facts place drops W0 rank, end-of-training VPD placement and carrying share (seeds disagree); VPD set re-switching inside its random range on 4 of 8 states.
