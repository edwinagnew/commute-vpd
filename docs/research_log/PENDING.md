# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 · decisions.md · decisions · remove card decision-counting-large-direction: ask 17 taken by default at 18:08, E70 Amendment 2 runs (largest direction with no covariance with the count, LOW first, then P and N).
- 2026-11-03 · experiments/TRIED.md · experiments · new row E70 Amendment 2, coded, not run: largest direction with no covariance with the count; the last amendment for E70.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e77-open-points.md · experiments card E77, claims table, sub-question status, timeline · new card from `notebooks/E77_open_points.ipynb`, figures `experiments/qwen_open_points/figures/`, row in TRIED.md: with all 196 matrices masked, small changes barely reach the logits at early blocks, reversing the lean on the own-output change; masks keep the output within about 0.5 nats of their own row, and dependence on tokens is not measured; all-masked is the extreme case; the lead's chat reading "masks carry the output" narrowed by the check.
