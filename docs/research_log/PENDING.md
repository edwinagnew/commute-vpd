# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · reports/analyses/2026-10-27_analysis_qwen-fresh-numbers.md · card decision-qwen-baseline (rewrite the headroom half), timeline · the fresh-numbers check ran (cluster job 39129, $0): the unchanged Qwen3-1.7B, same prompt and grader, gave identical answers to the baseline on GSM8K test, lost 0.039 ± 0.012 on GSM-Plus changed-number versions of the same problems (between the 0.02 ability and 0.05 memory cuts) and 0.008 ± 0.038 on GSM-Symbolic instances (too wide to decide); verdict ambiguous, as predicted. Rewrite the card's headroom half: option (a)'s check is done, the verdict is ambiguous, and the decision tree's next step, a prefix test on Qwen3-1.7B-Base (does it complete GSM8K test questions word for word as it does train ones?), is being planned before GSM8K is kept or dropped. Timeline: one line for 2026-10-27
