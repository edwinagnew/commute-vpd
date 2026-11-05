# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 · reports/analyses/2026-11-05_analysis_e99-definition-verify.md · E99's experiment card (new); the new-vs-reuse definition card / sub-question status; timeline · E99 stages 0-3 written up (notebook notebooks/E99_definition_verify.ipynb): the net share at a located pair reads 1 on a forced fine-tune and within band on a forbidden one on the copying pair, but the forced lowercase fine-tune caps at about 0.6 of the native gain, so lowercase stays "does not apply" and the batch decision holds (only the copying pair qualifies); the forbidden lowercase arm read below band after its stop rule ended training at step 5.
