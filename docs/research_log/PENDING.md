# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · reports/analyses/2026-10-27_analysis_qwen-gsm8k-baseline.md · next, timeline, decisions (new card: is 0.756 enough headroom, and drop or cap answers past 512 tokens) · Qwen3-1.7B GSM8K baseline (unchanged model, non-thinking, cluster L40, $0): greedy strict 0.756 (filed 0.60-0.75, narrow high miss; decision-tree "too little headroom" branch triggered by its letter, user to decide), lenient 0.783; 95.3% of answers within 512 tokens but those past it are mostly wrong (strict 0.32 vs 0.78), so drop-vs-cap for the held-out set biases it; 39% of sampled prompts mixed (31% lenient); 4.9% miss the strict format, mostly "The answer is \boxed{N}"
