# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-21 · parts/sections/decisions.html · decisions · add the card decision-wording-a-at-vpd (reports/decisions/2026-11-21_decision_wording-a-at-vpd.md): E247 (P1.47) finds the thin new edit reads 0.98 under wording A at VPD because 0.89 of its circuit is untouched and counts as reused, not because of the gains; the SAE order (0.57 below 0.69) rests on the same count; options (a) SAE primary, (b) change-weighting (inverts: thin 0.54, rebuilt 0.04), (c) keep A, report the untouched share beside R, state the limit (recommended); result page reports/analyses/2026-11-21_analysis_thin-edit-diagnosis.md
