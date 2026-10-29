# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-29 · reports/reviews/2026-10-29_review_fable-anchor2-grid.md · pending decisions (new card decision-e45-anchor2-grid, Qwen line) · anchor 2 learns only from step ~100 (own-row KL 4.5-5.3, maths KL ~0.6), above every filed grid point, so as filed it is unscored; its GSM8K accuracy 0.79 -> 0.13-0.20 looks like a one-line answer format carried over (answers 198 -> 18-47 tokens), which confounds its maths-row readings; Fable (advice): Amendment 2 = read anchor 2 at step 100 and match the other arms on the maths axis, KL stated; two $0 checks (GSM8K under a prompt at 100/300; teacher-forced loss on gold) separate damage from format; options and costs in the review; asks the user (ask 5)
- 2026-10-29 · experiments/qwen_updates/results/P.json, run.log · E45 card · stage P stopped: no prompt raises no-think GSM8K accuracy beyond noise (few-shot -0.033, recheck +0.016, givens +0.012; format template 0.680 vs 0.834 without), so anchor 1 (context distillation) has no prompt; decision for the user (card to follow)
