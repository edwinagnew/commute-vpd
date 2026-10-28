# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-28 · reports/plans/2026-10-28_plan_regating-line-reset.md (review question) · pending decisions (card decision-regating-reset: add the P0.4 confound question, options A/B/C, lead recommends A; Due to run: stage 0a now, 0b waits on it) · code review found the copy-task write adds the key's own embedding (tied embeddings)
- 2026-10-28 · decisions.md, parts (overview, questions, backlog, techniques, e31, e34) · same sections · the claim tests are E47, not E45 (E45 is the Qwen Q2 plan); text renamed in parts, republish needed
- 2026-10-28 E44 (qwen_maps) code written (50d9106), adversarial code review fixed (e5ebb33: resume kept stops, memory at batch 2, fp32 split storage, stage R rule, spend clock); first $0 rehearsal clean on pre-fix code (Gram bases match SVD to 1e-14); rerun on fixed code next, then the ~$8 H100 run. Card: due to run.
- 2026-10-28 E45 (qwen_updates, Q2) plan approved by the user with Fable's changes (reports/reviews/2026-10-28_review_e45-plan-fable-advice.md: packed invented facts, self-distillation null arm, seed 1 re-draws data); Job 1 code written (2e0927f), ~16 cluster debug jobs, $0; Job 2 gated, waits for E44. New card: planned / due to run.
- 2026-10-28 · reports/reviews/2026-10-28_review_e45-job1-code.md · E45 card · independent code review: nothing wrong in the planned order; deferred fixes (E resume reuses stale readings on a same-name T rerun; step-2 KL limit unscaled for own2; kv_tokens change mid-S double-samples)
- 2026-10-28 · reports/plans/2026-10-28_plan_qwen-q0q1.md (rehearsal record) · E44 card · $0 rehearsal on fixed code clean (Gram bases match SVD; stop, refusal, waive and resume paths as expected); ~$8 H100 run started
