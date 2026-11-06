# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 baseline-lead · TRIED row E117 (neuron_vpd_count): VPD's switched-on set and re-switching repeated on MLP neurons at VPD's count, copy/random maps and lowercase/ciphers; code and predictions filed (commits 19d2dcf9, d78511f4), job submitted by a pod-runner. Card follows with the results.
- 2026-11-06 baseline-lead · TRIED row E118 (qwen_comparator_table): E92's size, rank and steering and E102's carrying set on the saved Qwen states they left out, joined with every VPD row of E93, E94 (+Am1), E104 (+Am1) and E107 Am1; plan `reports/plans/2026-11-06_plan_e118-qwen-comparator-table.md`, existing cells joined in `experiments/qwen_comparator_table/results/table.md`, two debug jobs submitted. Card follows with the table (step 3).
