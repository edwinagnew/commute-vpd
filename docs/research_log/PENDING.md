# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 · reports/analyses/2026-10-29_analysis_e44-no-teeth.md, notebooks/E44_qwen_maps.ipynb (figures experiments/qwen_maps/figures/kept.png, ratios.png), reports/plans/2026-10-29_advice_e44-pass-test-FABLE.md · E44 card (add figures and notebook), decision card decision-e44-pass-test (v115 published it without options), timeline · STILL QUEUED after v115 (queued after the v115 flush started): Fable recommends masks from the original model for every split (a move from reading A to reading B: on Qwen reading A is passed by the masks alone, do-nothing included), as the last amendment, with a random control matched in KL to start; first a $0 layer-cut check (fine-tuned activations into Γ up to layer 8, 16, 24 or 28, then the original model) to tell whether the masks describe the change or copy the fine-tuned output from late layers. Card options: (1) masks from the original model, (2) a mask-cost or refit reading, (3) descriptive placement with no pass line. v115 already applied the result text (do-nothing 0.09 vs 1.42 nats, r2 0.975, no teeth) and the card with options A and B only.
