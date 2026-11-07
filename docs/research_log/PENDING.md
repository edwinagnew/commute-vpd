# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-07 regating-lead · reports/analyses/2026-11-07_analysis_e130-copy-protection.md (Amendment 1) · E130 card + claims (broad lowercase row) · E130 Amendment 1 (job 55031, L40, $0, controls exact): on Pile rows 256-511 the step-20 Adam copy reads m 0.78 (partial, 0.02 under 0.8, low power), so its reversal does not replicate; over four 128-row blocks its m is 0.62, 1.75, 0.60, 0.97 while the step-2,400 LoRA copy reads 0.23-0.28 on every block; "function-dependent" stands only as "not constant", placement adding excess is not established; synthesis reports/analyses/2026-11-07_analysis_67m-story.md updated to match.
