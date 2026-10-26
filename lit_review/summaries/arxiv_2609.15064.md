# What Does an LLM Learn from Reinforcement Learning? A Mechanistic Interpretability Perspective with Fixed-SAE Track

## Metadata
- **Authors**: Lingheng Du, Yiming Tang (equal contribution), Xufeng Duan, Dianbo Liu
- **Year**: 2026
- **Source**: arXiv preprint (2609.15064)
- **DOI/URL**: https://arxiv.org/abs/2609.15064

## Summary
The paper proposes Fixed-SAE Track: one sparse autoencoder (TopK) is trained per layer on residual-stream activations pooled from the base model and all RL checkpoints, then frozen and used to re-encode every checkpoint, so each latent index means the same direction throughout training. Applied to GRPO-style RL on math for Qwen2.5 models, it finds that RL-induced drift is small, gradual, late-layer, and mostly amplifies formatting scaffolds (answer wrappers, step breaks, operators, LaTeX delimiters), and that steering these latents into the base model reproduces much of RL's measured format-driven accuracy gain. The authors argue RL mainly elicits capabilities the base model already has, with a smaller genuine reasoning gain under sustained training.

## Key Findings
- Drift under GRPO is concentrated in about ten features; max feature drift grows with depth (0.013 at layer 6 to 0.060 at layer 27) and the active set turns over about 28% by the final checkpoint (Sec. 4.1, A.2).
- Amplified features are formatting/step scaffolds; suppressed features are more tied to problem entities and connective text (Sec. 4.2, Table 1). Sign patterns hold across GRPO, REINFORCE and RLOO to a rough degree (mean off-diagonal Spearman 0.42, A.3), though amplification is less reproducible than suppression.
- Adding the amplified directions to the base model at alpha=10 raises strict accuracy 0.050 to 0.192 and tag-rate 0.092 to 0.600, close to RL-final (0.175, 0.533); lenient accuracy changes only modestly (Sec. 4.3, Table 7). Random-direction steering shows no comparable peak.
- About 86% of feature amplification occurs in the first 50 steps; sustained constant-LR training (+2000 steps) adds a modest lenient gain (0.300 to 0.408) on top of a large format gain (Sec. 4.4, A.6).
- On 29 GSM8K problems that RL fixes, the base model solves all 29 under 64 samples (pass@64 = 1.00). The authors read this as RL raising the probability of trajectories already in the base distribution (Sec. 4.4, Table 2).
- A small arithmetic "reasoning" latent set R is causally necessary (ablation drops lenient GSM8K 0.358 to 0.242) but uniform injection recovers only about +0.06 of RL's ~+0.34 gain, so it is described as a dynamically sustained state, not a static direction (A.7).
- The format signature is descriptively present at 1.5B, 3B and 7B, but causal ablations are only clean at 1.5B (A.8). The qualitative format-up / content-down pattern is said to hold across SAE families, though exact feature identity is only claimed within sparse (TopK) SAEs (A.9).
- The intro promises a synthetic benchmark with features known by construction to test whether RL can create genuinely novel features. I could not find this benchmark or its results in the body or appendix of the text I read.

## Methodology
Residual-stream activations at the last non-pad prompt token (layer 27 primarily) are pooled over base and 13 GRPO checkpoints (steps 0-600, every 50 steps), standardized, and used to train a single TopK SAE (k=32, expansion 8, 12288 latents for the 1.5B model). The dictionary is frozen and each checkpoint is re-encoded. Drift is defined as the change in mean latent activation from base to final checkpoint, with firing rate and Spearman correlation to reward as secondary measures. Latents are interpreted by token-level strong-fire attribution. Causal tests use steering of an amplified latent set into the layer-27 residual stream with a two-grader protocol (strict, lenient, tag-rate), plus ablation of latent sets against matched random and format controls. Models are Qwen2.5-1.5B/3B/7B(-Instruct, -Math) trained with GRPO, REINFORCE, RLOO on GSM8K/MATH.

## Relevance to Proposal
**Score: 6**

The paper is the closest found so far to strand 2 (tracking features along training with a fixed dictionary rather than re-fitting per checkpoint, contrasted with SAE Track warm-starting). It is directly relevant to the question of whether a decomposition can stay interpretable along a post-training trajectory, and its "fixed basis, measure shifts in latent coordinates" design is the closest analogue of holding a parameter decomposition fixed while the weights change. It does not touch parameter-space decompositions (VPD/SPD/APD), LoRA/weight-space geometry, or the commuting map itself, and its conclusion (RL mostly elicits, with formatting dominating) bears on the proposal's "strengthen existing vs create new" framing only indirectly.

## Limitations
- Single-domain setting (math RL, GRPO-family) on small Qwen2.5 models; the main causal results are only clean at 1.5B (Sec. A.8).
- The fixed-dictionary premise is not validated against a reference: no comparison to independently fit per-checkpoint SAEs on whether the drift readings agree, beyond SAE-architecture agreement within families (A.9).
- Steering "recovers around 80%" of RL's gain as stated in the abstract is not what Table 7 shows. Steered alpha=10 exceeds RL-final on strict (0.192 vs 0.175), lenient (0.333 vs 0.308) and tag-rate (0.600 vs 0.533), so the figure as written needs checking.
- The pass@k result (Table 2) is computed on 29 problems selected for being base-wrong-greedy but RL-right, which biases toward problems the base model can sample correctly; the "all solvable" finding is weak evidence about capability boundaries.
- Lenient vs strict accuracy is used to separate format from reasoning, but the strict-lenient residual is up to 0.025 absolute (A.1, Table 3), which is comparable to some reported effects.
- The advertised synthetic benchmark for genuinely novel features is absent from the text I read, so the "can RL create new features" question is not answered in this version.
- Appendix A.12 notes an earlier steering analysis was superseded, so readers should rely on the revised dose-response.
- Feature attributions are based on token-level firing and interpretation labels assigned by the authors; no independent blind labelling is described.
- The appendix reproducibility note has a garbled pass@k estimator citation ("unbiased estimator of ?").

## Key Quotes
> "we introduce Fixed-SAE Track, a framework that trains one shared SAE per considered layer on activations pooled across the base model and all RL checkpoints, holding every feature direction fixed" (Abstract / Sec. 1)

> "Thus, RL does not globally rewrite the representation; it makes a focused edit concentrated near the model output." (Sec. 4.1)

> "A reasoning feature substrate does exist and is causally necessary, yet it cannot be injected as a static direction" (Sec. 5)

> "we find that RL-induced drift is small, gradual, concept specific, and concentrated in late layers" (Abstract)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
