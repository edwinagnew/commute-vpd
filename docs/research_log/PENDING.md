# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 · reports/analyses/2026-10-29_analysis_e44-no-teeth.md, experiments/qwen_maps/results/{C,S,D,K,score}.json · E44 card, pending decisions (new Blocking card decision-e44-pass-test), timeline 2026-10-29 · E44 rerun on an A100 ($3.66; $6.14 total): reproduction check held, stopped at stage K with no teeth at step 10. Γ recomputes the masks from each state's own activations, so any state holding the fine-tuned weights reads close to the fine-tuned model through the masks alone: do-nothing 0.09 nats vs 1.42 with the original model's masks (r2 0.975, plan expected ≈ 0); halves likewise; a random change sized like δW does not hurt (max ratio 1.0004). Not a code bug. Per the plan the line stops; the card asks the user how the pass test should change (Fable's options to follow in reports/plans/2026-10-29_advice_e44-pass-test-FABLE.md).
