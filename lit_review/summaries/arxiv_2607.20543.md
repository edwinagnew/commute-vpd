# When RL VR Shrinks the Reasoning Boundary: Diagnosing Pass@k Inversion

## Metadata
- **Authors**: Todd Y. Zhou (Harvard University)
- **Year**: 2026 (arXiv v1, 12 Jul 2026)
- **Source**: arXiv:2607.20543 [cs.LG]; framed as an ECCV workshop paper
- **DOI/URL**: arXiv:2607.20543

## Summary
The paper studies "pass@k inversion", where GRPO-style RLVR raises pass@1 but lowers pass@256 below the base model. The author claims the loss concentrates on "boundary" prompts, where the base model solves the problem only rarely but nonzero. A two-mode toy model explains this as finite-sample mode commitment. A prompt-gated KL anchor to the base model (PBA) is offered as an intervention probe. All evidence is behavioural (sampling statistics and token-level entropy). The paper never inspects weights, activations or circuits.

## Key Findings
All numbers below are the authors' claims, on Qwen2.5-7B trained with GRPO on Omni-MATH-Train.
- Omni-MATH-Test, base vs matched GRPO (Table 2): pass@1 10.2 -> 25.1; pass@256 69.1 -> 68.3. So GRPO is slightly below base at k=256.
- GRPO+PBA reaches pass@1 29.0 and pass@256 73.0 (Table 2). Three seeds.
- Regime study on 3000 prompts (Table 3): GRPO loses 654 +/- 35 boundary prompts that the base solved, and PBA loses 91 +/- 18. Boundary entropy retained is 19% for GRPO and 71% for PBA.
- Out-of-reach prompts gain essentially no new positives under either method ("out-of-reach positive unique remains 0.1"). The author reads this as RLVR not expanding support.
- Single-run controls (Table 5): global KL, entropy bonus, random protection, dropping protected prompts and early stop each recover only part of the coverage. Only the oracle-boundary protect row (p@256 74.1) is comparable to PBA (73.2).
- Gate ablations (Table 6) show that more frozen-base samples (G0=32) modestly help.
- Theory (Sec. 5): with w*(x) as the correct-mode mass, the chance that a GRPO group of size G has no positive is (1-w*)^G. All-zero groups give zero advantage, so rare correct modes get no reinforcement before shared-parameter sharpening drifts. Proposition 1 gives the sign of the pass@k change under idealized commitment.

## Methodology
Prompts are binned by frozen-base one-sample success p0(x), using 256 base samples, into four regimes. Solved-easy is p0>0.6. Boundary is p0<0.10 with 1-(1-p0)^256>0.4. Reachable is the middle band. Out-of-reach is coverage <=0.4. The author then compares base, GRPO and GRPO+PBA on pass@k at k from 1 to 256 and on base-solved -> trained-lost transition counts. PBA estimates w* from G0=8 frozen-base rollouts. Prompts with at least one success get ordinary GRPO loss. The rest get a token-level KL(pi_theta || pi_0) anchor with beta_a=1.0. Setup is AdamW, lr 1e-6, 1000 updates, checkpoints at 100/300/600/final, temperature 0.6, max length 8192.

## Relevance to Proposal
**Score: 3**

1. Evidence is purely behavioural (pass@k, per-prompt transitions, sampled-token entropy). Nothing looks inside the model.
2. Its result could hold while our mechanistic verdict goes either way. "The trained answer is already in the base sampling distribution" is the behavioural reading the project rejects. A prompt can sit in the base support (boundary regime) while the post-trained behaviour runs through a direction C that W0 does not linearly read out. Conversely, a "not through C" verdict is compatible with GRPO preserving coverage. The paper's claim that RLVR only reweights the base distribution (inference from Sec. 2/5, and the author's own statement; not tested mechanistically) says nothing about whether the weight update is carried by existing subcomponents. Its lost/not-lost split is about coverage, not mechanism.
3. Usable items (inference on my part):
   - The boundary / reachable / solved-easy / out-of-reach stratification by frozen-base p0(x) could label GSM8K prompts. We could then check whether early-step delta-W readings differ by regime. For Qwen3-1.7B on GSM8K most prompts would probably be solved-easy, so the boundary bin may be small.
   - The base-solved -> trained-lost transition count and Delta_cov could sit beside our readings as behavioural comparators.
   - GRPO-specific point: all-zero groups give no gradient. Early updates are therefore dominated by prompts with positives, which is a hint about what early-step delta-W encodes.
   - Run card has checkpoints at 100/300/600/1000 steps, though nothing about weights is reported at them.
   - Nothing is specific to Qwen3. The model is Qwen2.5-7B, on Omni-MATH, not GSM8K.
   - No controlled task with novelty fixed by construction.

The connection to the project is thin. It is a sharpening-literature paper that supports "RL reweights what the base has" only behaviourally.

## Limitations
- Behavioural evidence only. The "mechanistic" framing refers to a toy two-mode model, not to model internals.
- Authors' own caveats (Sec. 8): one model family, two math benchmarks, three seeds, single-run controls, a random-mask control with one mask, no checkpoint-level validation of the "collapse horizon", and no tuned regularizer sweep.
- Theory is an endpoint abstraction. Proposition 1 assumes the commitment of Eq. 8 and does not derive it from GRPO dynamics.
- Boundary regime labels depend on the thresholds, which the author calls not fundamental.
- My inference: the pass@256 gain from PBA over base (+3.9 points) sits uneasily with the claim that PBA only "preserves" base coverage. The paper does not explain this beyond the boundary transitions.
- Some numbers differ slightly between tables (PBA p@1 29.0 vs 28.8, p@256 73.0 vs 73.2/73.1; Table 3 "Boundary lost" 91 vs Table 11 "all lost" 73). I did not reconcile these. Text was PDF-extracted, so I trust the table cell values only as read.

## Key Quotes
> "RLVR without external exploration primarily reweights trajectories sampled from the student distribution." (Sec. 2, p. 3)

> "pass@k expansion requires increasing the correct-mode mass w*(x), while ordinary RLVR can only exploit, preserve, or collapse what the base distribution already contains." (Sec. 2, p. 3)

> "The mechanism is not that all-zero reward groups directly reward an incorrect answer." (Sec. 1, p. 2)

> "PBA does not discover solution modes absent from the base model" (Sec. 1, p. 2)

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv paper with no comments.

## Not read
I read the full extracted text (pp. 1-18). The paper's appendix artifact files (rollout caches, masks) and the cited papers (Yue et al. 2025, Zhao et al. 2025, Yuan et al. 2026) are not available, so I did not read them. Figure 1 is garbled in extraction; I used the tables only.
