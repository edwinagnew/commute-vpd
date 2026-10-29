# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 reuse-lead · `reports/analyses/2026-10-29_analysis_e55-stage-a.md`, `notebooks/E55_pcfg_anchors.ipynb`, `experiments/TRIED.md` E55 row, log figures `experiments/pcfg_anchors/figures/log_{labels,reproduction,controls}.png` (from `plot_log.py`) · E55 card (rewrite from this result: chip "mixed", Run/Found/Moved on; figures fig/e55_labels.png, fig/e55_reproduction.png, fig/e55_controls.png), claims table, Q5 status, timeline 2026-10-29 · Stage A ran ($0): the two checks work as a ground truth (controls clean, nulls 0); every fine-tune that learned to count b does not go through the count of a; the all-spurious fine-tunes go through it at 1K but read neither by 10K, so PA3 missed (not reversed); LOW possesses count b narrowly; only 2 of 11 labelled states are "existing", and whether Stage C runs on them is open
