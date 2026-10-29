# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
- 2026-10-29 · reports/plans/2026-10-28_plan_claim-tests.md (Amendment 1) · E47 run line · code written (t1-t5 + t1b, t2b, t2c, t4b, t5b, all ≤ 24 min L40 debug jobs); under review before submission
- 2026-10-29 · reports/plans/2026-10-28_plan_qwen-q0q1.md (Amendment record 2026-10-29) · E44 run line · E40's no-warm-up δW to be remade on a cluster L40 (experiments/qwen_maps/regen.py) and put on W&B, so E44 runs on any H100 without volume ydavm7ct4x; C reruns as the reproduction check; code written, not yet run
