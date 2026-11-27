# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-27 · fleet/paper/cells.tsv (cell P1.66) · next · New cell P1.66 for whip-lead, 2 runs, due 2026-11-28: Fig 2 on all 16 known-answer tasks: retained share at VPD, top neurons and SAE, three seeds
- 2026-11-27 · fleet/paper/cells.tsv (cell P2.90) · next · New cell P2.90 for regating-lead, 2 runs, due 2026-11-28: One re-gating share on both models: whole-answer log-odds form (corrected from "log-probability" by whip-lead), Qwen re-read
