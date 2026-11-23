# ADL and other §2 methods: GPT adversarial reviewer (2026-11-23, condensed faithfully; redteam HEAD e894f1ed, paper 4b33440)

Verdict: needs-attention. No ship as a published-method comparison. Relabel the ADL and steering scores end to end, document exactly what ran, and limit the negative conclusions to the tested variants and controls.

## ADL in the manuscript
- **Blocker, ours: sec4_setup.tex:39 calls the comparison "the activation difference lens of" minder2025narrow.**
  - The citation is now correct, but the name overclaims: the score is ours, using their direction and intervention, not their semantic lens evaluation.
  - The struck-out "crosscoder-based activation difference lens" with minder2025overcoming is still visible in main.pdf.
  - Fix: "our activation-difference intervention score, adapted from Minder et al."
  - sec4_setup.tex:53-54 lists the faithful rerun as future work.
- **Major, ours: main.tex:125-128 and sec5_results.tex:159-170 do not define the local quantity.** "Tracks the definition" and "Existing diffing tools cannot see reuse at fine-tune size" are too strong.
  - The saved values support only this: our score agrees in sign with the approximate write-share gap on 8/9 built pairs, while giving the new side the higher score on 7/9.
  - 0.73/-0.53 are E275's recomputation over the needed set; E266/E267 gave 0.70/-0.31. The procedural correlation uses 20 readable states.
  - Sources: adl/results/amend1/compare.json; claim2_needed/results/summary.json:1458-1464,1965-1968,1995-1998.
- **Pass, minor clarification: main.tex:186-195 describes their method correctly.** Add that they read the vectors with Logit Lens, Patchscope and steering. "Can report new structure" should say they detect representational differences, not mechanistic novelty.
- **Major omission, ours: appE_methods_detail.tex:86-102 and appG_departures.tex:47-58 never explain the scored ADL variant.** Fix: add the intervention, data, numerator, denominator, max and undefined rule, and the omitted evaluations.
- Bib: minder2025narrow (arXiv 2510.13900) is correct; minder2025overcoming is correctly the separate crosscoder paper.

## ADL code (src/commute/adl.py; experiments/adl/)
Kept from theirs: the position-wise mean of fine-tune minus base residuals on unrelated text; a middle residual (input to block 2); positive Logit Lens; §5 Eq. 1's rank-one replacement.
- **Major, ours: a different test and estimand.**
  - Theirs: per-direction increase in next-token CE on fine-tuning documents and pretraining text.
  - Ours: patching on task read windows, scoring only the answer token's log-odds, divided by the gain, max over five directions, no score if the gain is below 1. Positions 0-4 match their semantic default, not the causal sweep over 0-8.
  - Code: adl/run.py:182-213; holes_fill/run.py:89-91,120-130.
- **Major, ours: their semantic analyses are absent.** No Patchscope, coherence-calibrated steering, interpretability agent or LLM relevance grading. Exact answer-token membership replaces their grade. Code: adl.py:62-78; adl/run.py:200-217.
- **Major, ours: model, data, positions and update recipe.**
  - The source tests 33 organisms across 7 models; we use a 4-layer 67M on built tasks.
  - Directions come from 8,192 packed Pile-validation rows with no BOS, against their 10,000 FineWeb document starts.
- **Major, ours: a different null and aggregation.**
  - 64 random directions from Pile prefixes, and the max, not their mean.
  - Our Gaussian weight updates are not their Gaussian directions, and are divided by the paired state's gain. The do-nothing is undefined, not 0.
  - Copy failed stage 0 (0.0028 vs random max 0.0048); lowercase passed just above 0.05. This shows sensitivity to one shared direction only.
- **Minor, theirs:** a Fig. 7 / §5 Llama-size inconsistency, and toolkit defaults that differ from the written recipe.

## Other literature methods
- **Steering, blocker, ours: "This is the test of" Venhoff is false** (main.tex:926-933, appE_methods_detail.tex:90-91).
  - sec4_setup.tex:27's "at every block" is wrong: one selected block is applied (fair_steering.py:44-77).
  - Fix: call it "our mean-difference steering share". Turner, Rimsky and Ward are cited correctly.
- **Prakash / Vig / Syed, major, ours.**
  - main.tex:377-380 attributes the reverse patch direction to Prakash; appG:28-33 gets it right. Prakash §6.1 patches selected fine-tuned heads into the base and reads accuracy.
  - The full MLP patch is our neuron-level CMAP adaptation. The upstream-only variant, the half-gain count and the overlap are not all CMAP: do not relabel the whole family as appE:92-97 proposes.
- **Crosscoder (Minder / Lindsey), major, ours.** Correct citations, and the pinned trainer is called.
  - Qualify "trained by its authors' recipe" and "cannot see reuse" as an adapted run whose local readout failed its controls.
  - appE:98-101 should say "fine-tune-specific latent under their test", not "calls them new".
- **Jain / Hewitt-Liang, major, ours.**
  - sec5_results.tex:174-179's "ran the tests" needs the local adaptations stated (67M reverse training to base predictions with a KL half-life, adapted pruning, a Pile probe).
  - Minor: appG:20-27's "all inputs" should be the capability subdomain.
- **Lee DPO / Panigrahi, minor, ours.**
  - Scope Lee (main.tex:222) to GPT-2-medium toxicity, with the directions largely preserved.
  - Panigrahi's grafting copies existing parameter coordinates; keep our own grafts (new trained units) separate.
- **Geiger, major, ours: main.tex:287-301.** Adding a remainder gives reconstruction, not a bijection onto total settings: an SAE latent/remainder pair can decode to an activation the encoder would not send back.
  - Fix: specify the codomain and prove closure under interchange, or state the convention.
  - appG:9-19 should separate the older mean-ablation set from the current needed set. Minor metadata: pages 1-64.
- **Sutter, major, ours: main.tex:217-218 makes Sutter's theorem unconditional.** Theorem 1 is input-restricted, under Assumptions 1-5. Fixing the map from the base is our design response, not their implication.
- **VPD, major, ours: sec4_setup.tex:16-18 gives 38,912 components "each matrix"**; that is the total across 24. Its percentile wording is also wrong: each neuron's own 97th percentile on base Pile, upper tail only.
- **Others.**
  - BatchTopK supports the trainer, not our fixed transfer with a remainder.
  - IA³'s 3e-3 is from T-Few.
  - Roy-Vetterli supplies the statistic; the weighting is ours.
- **Other context checked.** LIMA, URIAL, Yue, Raghavendra, DeepSeek: fine (behavioural only). The Qwen RL appendix is not input by main.tex.
