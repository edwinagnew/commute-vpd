# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-09 · reports/analyses/2026-11-09_analysis_scorecard.md · method table section, timeline · Cell P2.2 filled: 67M scorecard from saved results, 14 task pairs and 3 do-nothing rows by six readings, 67 numbers and 35 not run, labels as built, no verdict and no combined score
- 2026-11-09 · reports/plans/2026-11-09_plan_definition-page.md · pending decisions, what is next · Definition page rewritten: recommends the frozen-body test (retrain only the last block of the original model, read on unseen values); one question open to the user; nothing run, no verdict until a dated decisions.md row
- 2026-11-09 · experiments/TRIED.md (E128 row), decisions.md · experiments, timeline · E128 paused by the user before any fine-tune: 105 facts a side built, nothing scored; the qwen-rl-lead line closes for the paper roadmap.
