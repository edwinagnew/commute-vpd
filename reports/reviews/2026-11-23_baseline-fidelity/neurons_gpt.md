# Neuron methods fidelity: GPT adversarial reviewer (2026-11-23, condensed faithfully; HEAD e894f1ed)

Verdict: needs-attention. No ship under the current attribution and the "fewest" wording. Sources checked: Vig (arXiv 2004.12265 §§3.2-3.3, 5.2), Syed (2310.10348 §§3-4), Prakash (2402.14811 §6), Goodfire "Router-Weighted Top-K Filtering", Cammarata essay.

## Neuron share and upstream half
- **Blocker, ours: not Vig's mediation estimate.** (main.tex:912-917; src/commute/neuron_r.py:102-119,133-144)
  - Vig: one fixed model; gender-edited vs unedited prompts; neurons at the profession-word position; single and grouped interventions; per-example proportional changes in the ratio of continuation probabilities.
  - Ours: identical prompts in two weight states. All post-GELU MLP firing at every block and position is transplanted into the base at once, and the ratio of mean log-odds change to mean gain is read. This is a joint cross-model sufficiency reading.
  - A high share does not isolate reused mechanisms.
  - Fix: relabel as "whole-width neuron CMAP, adapted from Prakash et al." (Prakash patches selected head outputs from the tuned model into the base). Keep Vig as context only.
- **Major, ours: the upstream half is a custom input-CMAP variant.** (neuron_r.py:68-91,108-119,133-144)
  - Each block's whole post-norm MLP input from the fine-tuned forward is fed to the base MLP. It can already carry the effects of changed earlier MLPs. It tests what base MLPs write on fine-tuned inputs.
  - Do not apply "CMAP at neurons" to the top-neuron share or overlap (appE_methods_detail.tex:92-96).
- **Major if pooled: E52 is an older variant, not the appendix's half-gain count.** (neurons.py:47-61; baseline_neurons/run.py:117-124,172-178; steering.py)
  - E52 ranks by absolute mean activation shift, uses log-probability, takes half the full gain on the grid [1,4,16,64,256,12288], and scores an infinite index on failure.
  - The appendix count traces to E98 and k_half_fill. Keep E52 separate.

## Neurons for half the gain
- **Blocker for an unqualified Syed/EAP label, ours.** (neuron_r.py:150-203; k_half_fill/run.py:88-99; realistic_reads.py:175-186)
  - EAP scores edges from paired clean/corrupt prompts and ranks absolute attribution. Syed allows nodes.
  - Ours ranks whole MLP neurons: base gradient x (base firing on fine-tuned MLP input - base firing), with signed products summed over positions and rows, sorted descending. It is a direction-of-improvement ranking.
  - Fix: label it "signed neuron attribution ranking of the upstream cross-model patch", with Syed credited for the Taylor approximation.
- **Major, ours: "fewest" is not computed.** (neuron_r.py:206-234; k_half_fill/run.py:173-201; method_table/plot_final.py)
  - Prefixes of one fixed ranking are tested at powers of two up to 8,192, plus 12,288; the other neurons are clamped to base. There is no search and no monotonicity, and ranking and scoring share the read rows.
  - The target is half the positive upstream-patch numerator. Undefined seeds are omitted.
  - The gap is log10(mean new / mean reuse). The twin cells are unpopulated.
  - Fix: rename to "first tested prefix reaching half the upstream-patch gain", and state the grid, the clamping and the omission rules.

## Top-neuron share
- **Major, ours.** (masked_reads.py:127-134,153-168; holes_fill/run.py:129-160,212-215; key_grid_baselines/run.py:131-159)
  - Goodfire: router-weighted absolute SwiGLU firing across 8 experts in Qwen3.6-35B-A3B, 320 of 4,096 kept, online mask; reads loss, KL and benchmarks.
  - Ours: dense GELU; masks from two unmasked forwards are applied as frozen masks with the complement zeroed. Value = (N01 - N00)/(T1 - T0), with a gain of at least 1 log-odds required.
  - Fix: call it a "cross-model top-set mask-transfer share inspired by Goodfire".
- **Major, ours: budgets and selection.** (plot_final.py:60-63,110-148; refit_96key/stage0.json; key_grid_baselines/plot.py)
  - The budgets are 64 per block (~2.1%) or the VPD on-count, not Goodfire's 7.8%.
  - The "wider gap" pick is the largest absolute gap on the scored seeds: a selected comparison that can change calls.
  - E244 plots both budgets separately and exports the whole-width r_n. Keep the provenances separate.

## Top-neuron overlap and the percentile featurization
- **Major, ours: overlap is a non-interventional mean layerwise Jaccard.** (neuron_overlap/run.py:73-96,129-145,192-201; stage0.json:194-204)
  - Answer position only, 240 per layer. Jaccard 0.5 means two-thirds of each set is shared.
  - The planted amplification failed against matched random on overlap (0.9623 vs 0.9652).
  - Cite "Neurons as an Interpretable Basis" for the ranking; Cammarata is context only.
- **Major if quoted: E155's own calls differ from the paper's recomputed calls.** For copy vs random lookup, every new seed exceeds every reuse seed (0.40809 > 0.40656): E155 says no call, the paper reader a reversal. The scorecard imports seven rows.
- **Major if conflated: E217 (topk_neurons) does no fixed top-K.** (topk_neurons/run.py; kstar.py; appG_departures.tex:41-46)
  - Per-neuron quantiles, both tails (p3/p97, also p10/p90 and p1/p99), pooled Jaccard with empty/empty = 1, remainders; overlap 0.231-0.361.
  - The paper's upper-only p97 comes from E229 (topk_upper/run.py:95-101,150-175), not E217.
  - Fix: separate the two percentile featurizations from the fixed-K baselines.

The saved readings can stay if the appendix names the actual interventions, the selection rules and the narrower conclusions.
