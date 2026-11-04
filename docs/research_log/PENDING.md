# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · `decisions.md` · pending decisions · remove cards decision-anchor2 and decision-e69-third-seed (user decided: TOFU; the two seeds as they stand) (qwen-rl-lead).
- 2026-11-04 · `reports/plans/2026-11-04_plan_e81-tofu.md` · pending decisions, experiments, timeline · new Blocking card (text in the plan's "Stage 0 under Amendment 1"): the unchanged model picks the right TOFU answer 41% of the time against 17% by chance; new E81 card; timeline line (qwen-rl-lead).
- 2026-11-04 · `decisions.md` · pending decisions, experiments, timeline · remove Blocking card decision-e81-chance (user chose A: keep TOFU, pass mark 0.588); E81 card and timeline line say training runs; if the card was never published, do not add it (qwen-rl-lead).
