# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · decisions.md · card decision-qwen-baseline · user chose (c), drop answers past 512 tokens; the headroom half stays open, rewritten to point at the RL-headroom/contamination look-up (reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md, in progress)
- 2026-10-27 · reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md · card decision-qwen-baseline (headroom half) · look-up: RL from 0.75 is a normal start (published gains a few to ~15 points, mostly format, reliability and borrowed thinking mode); Qwen contamination strong for Qwen2.5 on MATH, weak for GSM8K test, unmeasured for Qwen3-1.7B (one n=100 study: 9-point gap); options now: fresh-numbers check (GSM-Plus substitution + new GSM-Symbolic instances, cluster debug, $0) and score RL on fresh sets beside GSM8K test; risk: RL in non-thinking mode re-activates thinking, watch answer length
