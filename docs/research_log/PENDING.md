# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-29 · experiments/qwen_updates/results/check_format.json · pending decisions (card decision-e45-anchor2-grid: evidence) · the $0 check Fable proposed separates damage from format: under a few-shot prompt, the invented-facts models keep GSM8K accuracy near the untouched model (0.70-0.77 at steps 100 and 300, both seeds, vs 0.75), and their loss on the gold answers falls (1.20-1.22 vs 2.50); so anchor 2's unprompted drop to 0.13-0.20 is mostly a carried-over answer format, not lost maths
- 2026-10-29 · experiments/qwen_updates/results/S_own.json, T_own_s0.json · E45 card · own-answers set built (5,939 correct answers kept of 8,000, unprompted accuracy 0.834); the own arm's training stopped at its first-minute check (step-2 change on maths rows 0.0057 vs filed limit 0.005; step size inside its band); the null arm (do-nothing reference, also trained on its own samples) stopped at the same check (0.0067); both held for a decision (Fable advising)
