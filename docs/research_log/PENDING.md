# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 baseline-lead · E107 Amendment 1 planned and filed (882e5f02): VPD's switched-on set on the wrong-answer states against the known arm's, the VPD counterpart of E107's neuron set reading; one debug job, $0; plan `reports/plans/2026-11-06_plan_e107-amend1-vpd-sets.md`.
- 2026-11-06 baseline-lead · Decision card decision-neuron-baseline: remove it; ask 23 taken by default, option (a), E109 plans the neuron read at a larger update size (decisions.md last row).
