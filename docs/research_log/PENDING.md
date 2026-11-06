# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · decisions.md (two rows: reuse definition reset; adopted definitions frozen), reports/decisions/2026-11-06_decision_e99-reset-to-jain.md · Pending decisions (remove card decision-adopt-refined-rules: ask 1 dropped, superseded by the user's reset); E99 card ("Moved on": the rules are not adopted; the definition is decisions rows 81 and 87 only, Jain readout + held-out carrying + direction removal; old tasks lacked Jain's marker-plus-content shape; next E112, case-label pair); glossary/definition passages that quote E99's net share, admissibility, teeth, clause (ii) or bands as part of the definition: cut them · the definition is reset to Jain and frozen; E99's six rules dropped
- 2026-11-06 · experiments/TRIED.md (E112 row, coded not run), experiments/case_label/results/predictions.json · E112 card: the case-label pair coded and its predictions filed (real "reuses C", twin "not through C" under rows 81 and 87); not run.
