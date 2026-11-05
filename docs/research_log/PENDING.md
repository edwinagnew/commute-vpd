# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-05 · reports/plans/2026-11-05_plan_e98-neuron-r.md · timeline; the method-table / baselines card · E98 coded, not run: a neuron analogue of VPD's r on the 67M (headline r_half, both MLP matrices kept, beside the full-width r_n, d_n, i_n, e_n), on seven method-table rows, against the construction labels marked assumed; controls first in stage 0; $0 debug job `cluster/neuron_r.sbatch`.
- 2026-11-05 · reports/plans/2026-11-05_plan_e98-neuron-r.md · timeline; the method-table / baselines card · E98 Amendment 1: job 52747 stopped at stage 0's down_only control (d_n 0.4218 vs r_n 0.4250, missing the filed "d_n > r_n"); the rule was mis-specified, since training every down_proj makes later neurons re-fire and r_n counts that (0.425 of G, a blind-to reading). New rules in `predictions_amend1.json`: down_only records r_n, new down_last control (last block only: d_n = 1, rest 0).
