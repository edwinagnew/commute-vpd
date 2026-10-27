# Research log: pending updates

One line per event that changes what the log says, appended when the event's file is written:
`- YYYY-MM-DD · <file> · <log sections it changes> · <one line of what changed>`.
A flush applies every line to `parts/` (README.md → Layout), rebuilds, republishes, and moves the lines to the changelog row it writes, leaving
this list empty. Recipe: `README.md` → "Batched updates".
- 2026-10-27 · reports/reviews/2026-10-27_review_gamma-bidirectional-windows.md · gaps (or the affected cards' caveats), techniques · The 67M Γ attends both ways, so Γ on 41-token windows reads the 20 tokens after the trigger; between-model comparisons look safe (E32 ceiling 1.00-1.04), absolute window levels and rung-0 retained_ci are exposed; one ≤30-min cluster pass would settle it (for the re-gating/commute leads)
- 2026-10-27 · reports/analyses/2026-10-27_analysis_qwen-{loader,compute,findings-to-port,rl-gsm8k,task-choice}.md · next, timeline · Qwen3-1.7B line started (qwen-decomp-lead): five scoping reports; JAX checkpoint to be converted to fp32 torch with a ported Γ checked against a JAX reference; RunPod volume in US-MO-1; first experiment under discussion with the user (task and fine-tuning data)
