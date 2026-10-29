# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 reuse-lead · `reports/analyses/2026-10-29_analysis_e55-stage-a.md`, `notebooks/E55_pcfg_anchors.ipynb`, `experiments/TRIED.md` E55 row, log figures `experiments/pcfg_anchors/figures/log_{labels,reproduction,controls}.png` (from `plot_log.py`) · E55 card (rewrite from this result: chip "mixed", Run/Found/Moved on; figures fig/e55_labels.png, fig/e55_reproduction.png, fig/e55_controls.png), claims table, Q5 status, timeline 2026-10-29 · Stage A ran ($0): the two checks work as a ground truth (controls clean, nulls 0); fast fine-tunes never go through the count of a; the slow 80%-spurious (M3) and slow all-spurious (W1) fine-tunes go through it at 1K, and by 10K M3 has learned b (not through) while the all-spurious ones read neither, so PA3 missed (not reversed); a count-c-only answer is ruled out for W1 at 10K by a + b + c = 40, and a midpoint of count(a) + 1 and count(b) fits it but is untested (Amendment 1: per-row answers on the saved states); LOW possesses count b by the filed rule, narrowly, against uneven controls (shuffled labels only for b, NEVER also for a); only 2 of 11 labelled states are "existing", both at 1K, and whether Stage C runs on them is open; corrected after the independent check reports/reviews/2026-10-29_review_e55-stage-a-analysis.md
