# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 definition-lead · main page What's next (D3) + timeline 11-06 · Definition step 2 written (reports/decisions/2026-11-06_decision_definition-step2-qwen-anchors.md): under the frozen definition no Qwen anchor gets a reuse verdict (template 'does not apply'; facts and maths arms 'unreadable', expected from task shape, not measured); D3 keeps its built labels with the definition's reading printed beside; handed to qwen-decomp-lead.
