# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 baseline-lead · decisions.md row (ask 19), reports/decisions/2026-11-05_decision_new-vs-reuse-definition.md · overview definition box + glossary "Existing and new" · admissibility rule accepted by the user: C counts for a task only if g from W₀'s reading of C beats ignoring C on held-out inputs; constant-target tasks get input-dependent versions; edge cases filed first (E90). The box's "still open" sentence goes.
- 2026-11-05 baseline-lead · reports/plans/2026-11-05_plan_e90-new-vs-reuse-definition.md, decisions.md · overview definition box + glossary "Existing and new" · E90: definition frozen at rows 81, 115; one table of old labels by 11-10, two debug jobs; Qwen page by 11-07; the rest is apparatus. Three fixes: spared control, located mechanism or value directions with the hand write spared, answer = scored alternative. The box states plainly that the definition does not apply to emoticon, ladder, many cues and lowercase, with R7 as the untested repair.
