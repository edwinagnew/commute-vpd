# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-03 · reports/plans/2026-11-03_plan_e69-grpo-loop.md · experiments (new E69 card), pending decisions (Blocking: go-ahead on E69's cost line) · GRPO loop, null and own-samples SFT on Qwen rehearsed for $0 and reviewed; the H100 run waits on the user.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e71-scorecard-ls.md · experiments (new E71 card), timeline, decisions (new card decision-lowercase-ls-starts, `Question for you · commute-lead · new 2026-11-03`); text for all three in the report's "Log text for the clerk", figures experiments/scorecard_ls/figures/{check,keep_delta}.png · Lowercase least squares fails the check on Adam and, against prediction, passes on the rank-8 LoRA at one start; the card asks whether to re-read from more starts.
