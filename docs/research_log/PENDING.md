# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 · reports/plans/2026-11-06_plan_e105-rows-null.md · Qwen neuron baseline (what next, E103 card) · E105 coded, not run: the same-label, different-rows null for set identity, two disjoint halves of the 217 real rows at seeds 42 and 142, read at the matched step; filed rule label / rows / no call per row set, leans rows (real rows) and no call (TOFU rows); one $0 debug job, about 11 min.
- 2026-11-06 · reports/plans/2026-11-05_plan_e94-d3-rows.md (section "Amendment 1"), experiments/TRIED.md (E94, Amendment 1 row) · experiments page (E94 card: add "Amendment 1 coded, not run") · E94 Amendment 1 coded, E94's first of two: baseline-lead's known-facts fine-tune (same TOFU task, format and recipe as anchor 2, label "already known", seeds 42 and 142) through E94's reads, to test whether outside share, placement excess and think overlap follow the label (like the template fine-tune) or the task and format (like the TOFU facts fine-tune); matched size decides, the end state (norm 8.4 against 18.1) is description.
