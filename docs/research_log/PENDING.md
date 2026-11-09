# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-09 · decisions.md (E128 source row) · section 3 decisions, E120 card's next step · Remove the card "Anchor 2's backup: which source of post-cutoff facts?" (user chose A, Wikidata statements, with the reservation that it feels artificial and irrelevant for RL); E128 is being built.
- 2026-11-09 · reports/plans/2026-11-09_plan_new-decomposition-reruns.md, experiments/DECOMPOSITIONS.md · pending decision "move the faithful-map work to the new decomposition?" removed (user: yes, re-run decision-bearing experiments on new); timeline; state of play says which Qwen readings are on which decomposition
