# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-11 · experiments/disagreeing_pairs/results/amend1_stage0.json, stage0.json · E133 baselines card, next.html P2.17 row, timeline · P2.17 closed (E151, runs spent, no share read): new pairs where the prior and steering disagree could not be built on the 67M model; a cipher whose answers stay unrelated to the key could not reach a prior 0.5 decades above copy's (best 2.32e-4 against 2.35e-4 needed). So option b gave no reading; the paragraph on what the share adds is written from the nine built pairs.
- 2026-11-11 · reports/analyses/2026-11-11_analysis_p2-19-what-share-adds.md · E133 baselines card (what the share adds), next.html P2.19 row, overview, timeline · P2.19 filled: on the nine 67M built pairs the share is the only single reading that puts the reuse-built side higher on all nine, and its gap sizes follow neither the prior nor steering; but it adds no different call on direction (prior plus strength-1 steering reproduce all nine), pairs that set the two against each other could not be built, and the neuron readings separate few pairs. Whether the share or a cheap reading is right about reuse waits on scoring against part 1's labels (L2.2). No verdict on reuse.
