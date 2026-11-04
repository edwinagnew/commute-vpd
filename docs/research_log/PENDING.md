# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · `reports/analyses/2026-11-04_analysis_e81-tofu.md`, `notebooks/E81_qwen_tofu.ipynb` · experiments, timeline · E81 card links the report, notebook and two figures: neither rate answered, trained answers memorised, little carries to rephrasings; E81 ends there and its δW is not anchor 2. Card decision-e81-answered is decided (user chose a new plan, E84): do not add it, and remove it if already published (qwen-rl-lead).
- 2026-11-04 · `decisions.md` · pending decisions, experiments, timeline · remove card decision-e81-answered (user chose a new plan, training on rephrasings, E84, ask 7; E81 ends with its reading); E81 card says so and uses 33 and 31 min; it states the own question's edge over another author's question grew (0.131 to 0.194, 0.145) (qwen-rl-lead).
