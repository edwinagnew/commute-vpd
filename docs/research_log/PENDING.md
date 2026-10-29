# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
- 2026-10-29 · reports/plans/2026-10-28_plan_claim-tests.md (Amendment 1) · E47 run line · code written (t1-t5 + t1b, t2b, t2c, t4b, t5b, all ≤ 24 min L40 debug jobs); under review before submission
- 2026-10-29 · reports/plans/2026-10-27_plan_broad-fragility.md (Amendment 3) · E39 card run line; decision-lora-sgd outcome · E39 Amendment 3 written (SGD teacher-trained decompositions on the second rows, stop if both pass; labelled SGD at step 1,000; rank-8 LoRA trained by plain SGD, E27's grid 0.03-3, teacher-trained and read; ~$1.8 H100); under code review before the run
