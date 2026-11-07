# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
A line starting `HOLD ` (not `- `) stays in `PENDING.md` through a flush: use it for a line whose report was found wrong, in place of removing it. It reads `HOLD YYYY-MM-DD <role> · <why> · <the line>`; a clerk skips it, and the role that held it restores or deletes it (friction 317).
- 2026-11-06 qwen-rl-lead · decisions.md (anchor 2 after E120) · E120 card + timeline · Ask 13 taken by default: E128 (post-cutoff facts) is the backup for anchor 2; E120 SFT arms run on the 77-a-side pools as a provisional row (cluster, $0); E128 plan reports/plans/2026-11-06_plan_e128-post-cutoff-facts.md awaits the facts-source answer.
- 2026-11-07 qwen-rl-lead · experiments/qwen_pass_pools/results/train_known_s0.json, read_known_s0.json · E120 card (running) · FILED PREDICTION MISSED (P4, seed 0): SFT on the 62 known-pool problems for 100 steps (103 passes) drops GSM8K test strict from 0.784 to 0.422 and raises held-out gold loss 2.01 to 2.90; E61 gold arm at the same recipe dropped to 0.62 on many more problems. Read as overfitting, cause not named; seed 1 and new arms running.
