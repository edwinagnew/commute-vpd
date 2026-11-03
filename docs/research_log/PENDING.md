# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.
- 2026-11-03 (qwen-decomp-lead) Card `decision-d1-test`: remove, decided (ask 24, `decisions.md`): finding on masked tests filed, clean-model table is D1's table, teacher training with masks in its loss is the map. New card E74 follows with its report.
- 2026-11-03 · `reports/analyses/2026-11-03_analysis_e72-d1-second-test.md` · new experiment cards E72, E73, E74; sub-question status for D1 (a map on Qwen3-1.7B that keeps the decomposition faithful for a real update, judged by a test that can fail a bad placement); claims table; timeline · Under all 196 matrices' masks, fixed or stochastic, the whole update shifts the logits by 0.03-0.13 of its clean amplitude, about what 1-3% of it does clean; masks made on the fine-tuned model move gain by at most 0.04; why is not shown. Clean table: the in-span/remainder gap is about what size gives; filed label "carried" stands as filed. Independent check holds both with wording changes, no bug.
- 2026-11-03 · `experiments/TRIED.md` · rows E72, E73, E74 · Added. Not yet read: E77 (early control, partial routing, displacement), E75's random slice.
