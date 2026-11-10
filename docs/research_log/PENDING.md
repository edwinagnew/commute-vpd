# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-10 · reports/analyses/2026-11-10_analysis_gate-page-working-definition.md, reports/decisions/2026-11-10_decision_gate-working-definition.md, experiments/old_task_labels/results/labels_cut_0p8_0p5.json · decisions section (remove the pending-decision card for ask 29, id decision-gate-working-definition), P1.1 gate status, changelog · The P1.1 gate is decided (user): the second wording of reuse is adopted tentatively as the working definition, to be revised if it cannot detect built; cut-offs rounded to 0.8 for the re-weighting share and 0.5 for the rewiring share (from 0.794 and 0.476). Re-labelled from saved results, one label moved: the 300-step ordinary copies of the hand-made Always on change now read no map found (seed 2 rewiring share 0.48), not rewired; no old 67M task moved (only the real case label reads re-weighted; old-task reuse sides other than lowercase keep rewiring shares of 0.58 or more). The within-pair comparison is unchanged; the built control runs as a new cell.
