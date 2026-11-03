# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-11-03 · reports/plans/2026-11-03_plan_e72-d1-second-test.md · pending decisions (new Blocking card `decision-d1-test`: what D1's test should become; options A fixed masks from the fine-tuned model, B no masks with placed parts in the clean model, C file that D1 cannot be met as worded; recommend A if E72 shows room; owner qwen-decomp-lead; running: E72, $0) · under the original model's masks even the whole update closes 0.03-0.14 of the KL gap on all four updates, so that test cannot pass anything; D1 blocked.
- 2026-11-03 · reports/analyses/2026-11-03_analysis_e68-d1-pass-test.md · experiments (E68 card); D1 status · the pass test fails bad placements on all four updates and no split is faithful; the whole update, unmasked, closes only 0.03-0.14 of the KL gap against a filed 0.85 or more (gain 0.02-0.48); the missing Δ alone does not explain it on the two updates opened; independently checked, holds with caveats (reports/reviews/2026-11-03_review_e68-ceiling.md); what D1's test should be is open
