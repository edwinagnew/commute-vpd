# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · `reports/analyses/2026-11-04_analysis_e81-tofu.md`, `notebooks/E81_qwen_tofu.ipynb` · experiments, timeline · E81 card links the report, notebook and two figures: neither rate answered, trained answers memorised, little carries to rephrasings; decision stays on Blocking card decision-e81-answered (qwen-rl-lead).
