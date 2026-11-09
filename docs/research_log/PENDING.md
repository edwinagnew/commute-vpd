# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-09 · reports/plans/2026-11-09_plan_definition-page.md · pending decisions, part 1 · Definition page for P1.1 written, waiting on the user. Prakash et al. does not switch a part off, does not find it away from the task and has no twin, so the plan's attribution changes. Remove card decision-definition-task-search (ask 11 dropped, superseded by the 2026-11-09 plan).
- 2026-11-09 · reports/analyses/2026-11-09_analysis_p1-2-e62-reread.md · part 1 · E62's copy and lookup pair re-read under the proposed wording, provisional until adopted: copy seeds 0 and 1 read reuse, seed 2 not through copying at 0.471, all three lookups not through copying. No new run.
