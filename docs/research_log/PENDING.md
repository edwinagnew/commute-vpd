# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 qwen-rl-lead · reports/plans/2026-11-06_plan_e120-pass-rate-pools.md (Amendment 2) · E120 card + timeline + new pending-decision card "anchor 2: maths pools too small, go to the facts backup?" · Grader fix ran ($0): 14 of 16 named misses now match, controls pass, but the fixed problems were ones the model nearly always solves, so they moved to known; pools 77 a side (< 100), so the plan's branch goes to the backup (post-cutoff facts) after a decision card.
- 2026-11-06 · experiments/neuron_vpd_table/results/full_table.md · neuron-vs-VPD table (section 7, anchor neurontable) · one line under the E93 ordering table: E93 Amendment 1 (scaled adjoint split) fails all eleven Qwen states, so no column can order it; no new row, comparators unchanged.
- 2026-11-06 regating-lead · reports/analyses/2026-11-06_analysis_e127-ls-start-faithfulness.md · E127 card + timeline · The least-squares start's excess over do-nothing under the adversarial check is all placement on both tasks (its own weight change kept in Δ reads better than do-nothing: lowercase 2.95 vs 4.79, start 8.33), seen only by the adversary and removed by holding the alive masks at 1, but the hold also changes the adversary (W₀ 1.60 vs 0.69 nats) and flattens a state with its change in Δ, so do-nothing under the hold (Amendment 1) decides whether that reading is specific.
