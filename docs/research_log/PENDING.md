# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-04 · `decisions.md` · pending decisions, experiments, timeline · remove Blocking card decision-e81-chance (user chose A: keep TOFU, pass mark 0.588); E81 card and timeline line say training runs; if the card was never published, do not add it (qwen-rl-lead).
- 2026-11-04 · `reports/analyses/2026-11-04_analysis_e83-logit-shift-67m.md` · experiments, timeline · new E83 card: on the 67M masks shrink the whole update to 0.15-0.57 of its unmasked size (Qwen 0.10-0.26); controls pass; a one-off for Q3, re-gating line not reopened
- 2026-11-04 · `reports/plans/2026-11-04_plan_e81-tofu.md` · pending decisions, experiments, timeline · new Blocking card decision-e81-answered (text in the plan's "Read under Amendment 2"); E81 card rewritten to the read (qwen-rl-lead).
