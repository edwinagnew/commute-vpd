# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 qwen-rl-lead · docs/research_log/parts/sections/decisions.html · pending decisions (new card decision-rl-extra-reads, already written in parts; intro line) · Card added, waiting on readings, not on the user: a second seed of own-samples SFT runs only if a conclusion rests on that arm's gap to RL or the null; seed 142's crossing only if a TOFU reading changes between its first past-line state and its full update. qwen-rl-lead finished until then.
