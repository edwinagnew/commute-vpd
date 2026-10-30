# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 baseline-lead · reports/analyses/2026-10-29_analysis_e52-baseline-neurons.md (section "Amendment 1"), notebooks/E52_baseline_neurons.ipynb (Amendment 1 cells), experiments/TRIED.md (E52 amend 1 row) · E52 card, timeline · E52 Amendment 1 ran (job 45898, $0): with up to 31 examples next to the query the original model recovers 0.22 of the copy gain vs 0.18-0.19 cipher, top-1 0 everywhere, induction control 0.82 passes; P4a fails as leaned, copy label is "mechanism only" (never behaviourally verified); departures from Olsson et al. 2022 and Lin et al. 2023 listed.
