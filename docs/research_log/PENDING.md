# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · decisions.md, reports/plans/2026-10-28_plan_qwen-rl.md · decisions (remove card decision-qwen-baseline; decided: keep GSM8K, score on GSM1k), Q9, next (Qwen card: Due to run E40), timeline · user's go-ahead: RL from 0.756 is worth it for this project (signal on 39% of problems, small steps, a strengthening update); E40 plan: control on maths text, RL 100 steps, shuffled-reward null, SFT on human solutions
