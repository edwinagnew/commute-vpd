# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · docs/research_log/parts/experiments/e144.html · experiments (new E144 card); table of work (P3.2 filled); timeline · P3.2 filled, surprise: the log-odds re-gating share on Qwen (post-hoc deviation from E132's KL share) is unreadable: every update and random change lowers the trained answer's log-odds on its own rows; provisional, no verdict; publish needs fig/e144_gains.png
- 2026-11-10 decisions: remove the pending card "On Qwen3-1.7B the re-gating share sits near its ceiling..." (decision-qwen-regating-share). User chose option (b) (decisions.md row 2026-11-10 "Qwen re-gating read in log-odds as well").
