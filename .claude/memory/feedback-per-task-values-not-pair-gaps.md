---
name: feedback-per-task-values-not-pair-gaps
description: "Figures show each reading's value per task (reuse tasks vs new tasks), not pair gaps, tallies or cell text; plain matplotlib, little text"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2e4783ac-45db-47c9-85e5-89de66777986
  modified: 2026-11-24T04:03:35.025Z
---

Show each reading's value on every task, tasks built to reuse beside tasks built new, rather than gaps between built pairs; no agree/reverse/no-call tallies, no numbers printed in cells, minimal panel text. Matplotlib (seaborn-like look; seaborn is not in the project env).

**Why:** user, 2026-11-23, on Fig 3(a): "i've said this so many times but comparing in pairs seems useless when you could just say the numbers for each of the 8 tasks"; the tally column "looks terrible"; "way too much text". Also Fig 4(a) "too much info".

**How to apply:** any new scorecard or baseline figure (P2.72/P2.73 included): one small panel per reading, per-task means with seed ranges, colour by built label; per-pair calls only in the appendix. See [[feedback-plots-over-tables]].
