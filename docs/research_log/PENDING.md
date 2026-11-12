# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
A line `CONSENT YYYY-MM-DD <role> · <the user's words>` records that the user waived the main-page word cap for this flush (README.md → Main-page length; friction 688).
- 2026-11-12 · reports/analyses/2026-11-12_analysis_p3-6-qwen-cheap-readings.md · Part 3 / Qwen experiment card (E162, E166), timeline · P3.6 filled (cluster, $0): only effective rank separates RL from its shuffled-reward null (1.5 seed ranges); the top-neuron overlap separates neither arm contrast, though all 19 Qwen updates keep fewer top neurons than a size-matched random update (RL 0.89 vs 0.997). Overlap is description only; no reuse/new label. Premise check doubtful: rank may follow update size.
- 2026-11-12 · reports/decisions/2026-11-11_decision_qwen-overlap-floor.md · Pending decisions · remove the card: user chose four floor draws 2026-11-12 (decisions.md row)
