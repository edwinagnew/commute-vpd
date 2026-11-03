# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 · experiments/TRIED.md · experiments card E78 · new row, not run yet: anchor 1 retrained at E61's rate, δW in fp32 at steps 1, 2, 3, 5, 10, 20, 50, 100 (`reports/plans/2026-11-03_plan_e78-anchor1.md`); the card waits for the run.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e69-grpo-loop.md · experiments card E69, claims table, sub-question status, timeline · report and `notebooks/E69_qwen_rl.ipynb`, figures `experiments/qwen_rl/figures/`, TRIED row rewritten: RL beats the shuffled-reward null on the same problems by +3.2 to +7.4 (se 1.0-1.1) on GSM8K test, about +5 inside the window with format ignored; the null's filed miss (+5.6) mostly cancels inside the window, its sampled collapse is the answer line, the cause of its whole-answer gain is open; one null seed.
