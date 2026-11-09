# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-09 · experiments/qwen_regating_new/results/summary.json · table of work (P3.1, filled) · Re-gating share per saved Qwen update on the new decomposition beside the old one: all six states near the ceiling (0.991 to 0.9992), the step-10 fine-tune and gold separate from the two anchor 2 seeds as on old, by under 0.005; random changes read high too, so the share has little room; owner qwen-lead
- 2026-11-09 · reports/decisions/2026-11-09_decision_qwen-regating-share.md · pending decisions (new card, Soon) · Add the card from this file: whether the re-gating share stays as part 3's reading on Qwen, given random changes read near the ceiling, owner qwen-lead
- 2026-11-09 · decisions.md and reports/plans/2026-11-09_plan_reuse-variable-first-test.md · definition section, timeline · The user adopted the second wording of reuse (a split at a variable, labelled by how the fine-tune sets it) for testing, with "computed" meaning linearly readable; no verdicts on saved runs yet. A first test on four pairs with labels known by construction is proposed, nothing has run.
