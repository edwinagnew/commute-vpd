# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · decisions.md, reports/plans/2026-10-28_plan_qwen-sft.md, reports/plans/2026-10-28_plan_qwen-rl.md · decisions (remove card decision-qwen-baseline; decided: keep GSM8K, score on GSM1k), Q9, next (Qwen card: Due to run E40), timeline · RL from 0.756 judged worth doing (plan held, `2026-10-28_plan_qwen-rl.md` §1), but the user chose SFT and the existing decomposition first: E40 = control on maths text, SFT on human solutions, do-nothing and teacher training at steps 1, 10, 100, states saved for later maps
