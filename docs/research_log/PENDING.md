# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-10-30 · experiments/performed_anchor/results/check_amend1.json, decisions.md (last E53 row) · E53 card (Amendment 1 line), timeline · E53 Amendment 1 W0 check stopped by its filed rule: the original model does not lowercase 50 title-case words inside cased text (-4.72 nats, 90% -5.56 to -3.95; repeats -4.41). User chose: broad cipher alone + E53 re-read with true off-target KL.
- 2026-10-30 · reports/plans/2026-10-29_plan_performed-anchor.md (Amendment 1b), experiments/TRIED.md (E53 Amendment 1 and 1b rows), experiments/performed_anchor/results/predictions_amend1b.json · E53 card · Amendment 1 W0 check stopped (narrow lowercase −4.72 nats, not performed); Amendment 1b filed: broad cipher alone, gate = at least as broad as E53 lowercase on the true off-target KL, both ratios ≤ 0.5 = breadth does not explain E53 gap; not run yet.
