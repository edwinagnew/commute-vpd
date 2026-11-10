# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · reports/decisions/2026-11-10_decision_old-task-cutoff.md · pending decisions (remove card decision-old-task-cutoff); decisions · ask 28 answered by the user: option (c), the gate page carries the re-weighted share within matched pairs, absolute cut-off listed as open; decisions.md row added.
- 2026-11-10 · reports/decisions/2026-11-10_decision_gate-working-definition.md, reports/analyses/2026-11-10_analysis_gate-page-working-definition.md · pending decisions (add card decision-gate-working-definition, no recommendation: gate decision); table of work (P1.1 row: gate page written, figure experiments/old_task_labels/figures/gate_within_pair.png) · gate page for the second wording written: re-weighting share ranks every reuse side above its new twin in 5 of 5 matched pairs, only the real case label clears the 0.794 cut-off, built never read, built control designed not run; the user is asked to adopt, adopt with the cut-off as a follow-up cell, or reject.
