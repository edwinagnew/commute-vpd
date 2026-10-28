# Review: Fable's advice on the E45 (Q2) plan

*2026-10-28. Fable advisor, reviewing `reports/plans/2026-10-28_plan_qwen-q2.md` (704f2ae) before it goes to the user. Advice, not a decision.*

**Verdict: go with changes.** Job 1 now ($0); Job 2 stays gated and waits for E44. All changes are $0.

1. **Anchor 2 cannot be learned at the 0.1 save.** The warm-up curve (`experiments/qwen_sft/results/amend1_rehearse_1/F1.json`) crosses 0.022 at step 3, 0.11 at step 5, 0.36 at step 10, whatever the data; one 60-token fact per row means only 320-640 of 3,600 sequences seen by step 5-10. Pack ~8 facts per 512-token row (or ~100 people); read `anchor2_learns` at seed 0 before other anchor runs.
2. **Anchor 1 and own samples are nearly the same operation** (SFT on the model's own T=0.7 samples). Add a self-distillation null arm: SFT on unfiltered unprompted samples (already drawn; one 10-min job). Anchor 1 − null isolates the prompt, own − null the filter. Confound sentence: "shares its domain and its data generation". State beside the format fallback that it makes anchor 1 a format fine-tune.
3. **Route split with Δ off** drops O(δW) on attention in all four cells: report K11 − K₀ beside the share; unscore when it exceeds half of K00 − K₀. Say whether the route split runs in Job 2 or arrives in Q3.
4. **Grid.** Save the bracketing step too and read at the state nearest each grid value, stating the matched KL. For gold the grid sits in the warm-up ramp; step 100 serves only behaviour predictions.
5. **Scope.** Think-overlap row and 5th template are Q3 material: name, don't file.
6. **Open questions.** Job 1 ungated: agree. Job 2: cluster L40 `normal` if E44's peak fits 46 GB, else pod with stage T. CD via SFT on samples: accept with the near-identity caveat. Seeds by batch order only: no — seed 1 re-draws the data (anchor 1: 8 samples per problem, split; anchor 2: second name draw). Rate raise to 2e-5: keep; unlikely needed.

**What would change it:** anchor 2's own-row KL rising far slower than gold's; E44 showing K11 − K₀ small on attention; the user ruling a null arm out.
