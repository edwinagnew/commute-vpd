# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · experiments/qwen_sft/results/G.json · decisions (new card decision-qwen-maths-adversary, Soon: E40's control G read "ambiguous" against an expected pass: on the model's own GSM8K answers the importance-masked KL is 0.059 (web text 0.142, reproduced exactly) but the adversary reads 4.24 nats against web text's 1.70 (reproduced), 2.5×, between the filed 2× pass and 3× fail lines; how should E40's maths-row adversary readings (do-nothing and teacher training at step 100) be read: as a ratio to 4.24 (recommended), or as a sign the decomposition is fragile on task text) · Q9 · G is apparatus; the run continues by its filed rule
