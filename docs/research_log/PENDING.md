# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · decisions.md · card decision-qwen-baseline · user chose (c), drop answers past 512 tokens; the headroom half stays open, rewritten to point at the RL-headroom/contamination look-up (reports/analyses/2026-10-27_analysis_qwen-gsm8k-rl-headroom.md, in progress)
