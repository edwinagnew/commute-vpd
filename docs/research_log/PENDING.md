# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
Any lead's clerk may flush any lead's lines, whoever holds `fleet/log-lock`. So a clerk takes the lock, pulls, then
reads this file afresh and flushes what it holds then; a line its brief named that is gone was flushed by another
clerk: find it in the changelog rather than re-adding it (friction 84, 117).
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving this list empty.

- 2026-10-29 · experiments/TRIED.md (E53 row), experiments/performed_anchor/results/predictions_a0.json · E53 card (ongoing), timeline · E53 stage A0 coded and filed (W0 only: lowercase ≥ +1, cased ≤ −1, cipher ≤ −3 nats; stop if control misses, lowercase ≤ 0 or cipher > −1); not run yet

- 2026-10-29 · experiments/performed_anchor/results/a0.json (RunPod 4090, $0.08) · E53 card (new, ongoing), timeline · E53 A0 GO: the original 67M prefers lowercase forms in lowercased text by 1.18 nats (90% 0.84-1.56, 6,070 positions), capitals in cased text by 5.58, and cannot do the 50-word cipher (-5.53); control -3.14576. Fine-tunes wait on plan questions 1-3.

- 2026-10-29 · reports/analyses/2026-10-29_analysis_e44-amend1-layer-cut.md, notebooks/E44_qwen_maps.ipynb (Amendment 1 section), experiments/TRIED.md (E44 Amendment 1 row), experiments/qwen_maps/figures/amend1_layer_cut.png · E44 card, pending-decision card on the pass test (add as evidence for Fable's option 1), timeline · E44 A1 layer cut (cluster 44065, L40, $0): controls hold; the first 16 layers' fine-tuned weights make 93% of the update's output effect, yet Γ fed those layers' fine-tuned activations recovers none of the masks' reproduction (m(16) = −0.0025 vs u(16) = 0.93); it appears only with the last taps (m(24) = 0.31, taps alone 0.99, readouts alone 0.22). Verdict "copying", carrier taps: do-nothing's pass on reading A is Γ copying the fine-tuned output. Input to the user's choice of option 1 (reading B, Amendment 2), not a decision
