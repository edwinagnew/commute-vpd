# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 baseline-lead · decisions section: remove cards `decision-stable-neuron-ranking` and `decision-invented-facts` (asks 25, 26 closed "retargeted, not taken", `decisions.md` last row). Timeline 11-06: one sentence, "Baseline retargeted by the user: neurons and cheap comparators beside VPD's existing readings (67M re-gating states, then Qwen's E93/E94/E104 rows), one table by 11-14; E115 and E109 Amendment 1 dropped." TRIED rows E109 Amendment 1 and E115 (dropped at plan).
