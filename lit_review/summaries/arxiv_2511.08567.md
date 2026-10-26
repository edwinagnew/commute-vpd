# The Path Not Taken: RLVR Provably Learns Off the Principals

## Metadata
- **Authors**: Hanqing Zhu, Zhenyu Zhang, Hanxian Huang, DiJia Su, Zechun Liu, Jiawei Zhao, Igor Fedorov, Hamed Pirsiavash, Zhizhou Sha, Jinwon Lee, David Z. Pan, Zhangyang Wang, Yuandong Tian, Kai Sheng Tai (Meta AI / UT Austin)
- **Year**: 2025 (arXiv 2511, November 2025)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2511.08567

## Summary
The paper argues that the apparent sparsity of RLVR (RL with verifiable rewards) weight updates is a surface effect of a stable, model-conditioned optimization bias: for a fixed pretrained model, RL updates land in the same parameter regions across runs and datasets, and bf16 storage makes the sub-ULP part of that bias look like sparsity. It proposes a "Three-Gate Theory" (KL anchor, model geometry steering updates off principal singular directions, bf16 precision as a visibility filter) and supports it with spectral and mask-overlap measurements, a geometry-scrambling intervention, and two PEFT case studies (sparse masks and LoRA vs PiSSA).

## Key Findings
- Update sparsity is high for RLVR (36–92% of bf16 weights unchanged) and low for SFT (0.6–18.8%), but the sparsity largely disappears when the learning rate is raised so sub-ULP updates become representable, so it is a readout of the bias rather than its cause (Sec. 2.1, 2.3).
- Update locations are consistent across five RLVR runs with different data and algorithms (Jaccard well above the Bernoulli baseline; stripe-like row/column consensus patterns in Q/K/V/O), and the pattern emerges early and persists over training (Sec. 2.2).
- RLVR preserves the top-k singular subspaces and spectrum of the base model (small principal-angle rotation and normalized spectral drift), while SFT rotates and distorts them more (Sec. 4.1), and the same holds for agentic and DPO/SimPO checkpoints (Sec. 4.4).
- RL updates have sub-random overlap with "principal" weights (top-magnitude entries of the rank-k SVD reconstruction) and super-random overlap with low-magnitude weights (Sec. 4.2).
- Scrambling geometry in layers (orthogonal rotation / head permutation) of Qwen3-4B-Base collapses update-mask overlap to chance in those layers, which the authors read as causal evidence that pretrained geometry drives the bias (Sec. 4.3).
- A fixed mask built from the pretrained weights (exclude principal, include low-magnitude; about 70% of parameters) tracks the dense RLVR KL trajectory and accuracy, while principal-only masks learn worst (Sec. 5.1). PiSSA gives no gain over LoRA and collapses earlier at high learning rates (Sec. 5.2).

## Methodology
Empirical weight-space analysis of publicly released RLVR, SFT, agentic and RLHF checkpoints (mostly DeepSeek-R1-Distill-Qwen-1.5B, Qwen3-8B, Qwen3-14B, Llama-3, Qwen2.5-7B). Diagnostics: a bf16-aware "unchanged weight" probe, Jaccard and consensus ratios over update masks, principal-angle and normalized singular-value drift, and overlap of update masks with a rank-k principal mask and a low-magnitude mask against random baselines. Causal test: orthogonal rotation and head permutation of selected layers. Theory: a one-step KL-leash argument (Prop. 3.1–3.2) and Wedin-type subspace perturbation bounds (Thm. 3.3, Cor. 3.4), mostly in appendices. Interventions: sparse fine-tuning masks and LoRA/PiSSA sweeps (ranks 8/32/64, three learning rates, 200 steps) on DS-Qwen-1.5B and Llama-3.2-3B.

## Relevance to Proposal
**Score: 5**

The paper is about where RL updates δW land in weight space relative to the top singular subspaces of W0, which bears on the proposal's strand 3 (structure of post-training updates: alignment of δW with pretrained singular vectors, sparse fine-tuning, LoRA geometry) and strand 4 (RL sparsity of updates). It offers a concrete, testable claim that RL steps avoid the principal subspace, which a commuting map Φ(D, δW) would need to respect, but it says nothing about parameter decompositions (VPD/SPD/APD), subcomponent faithfulness, or decomposition stability, so it is a background input rather than a method source.

## Limitations
- The principal and low-magnitude masks are proxies for curvature; the authors acknowledge that direct curvature is intractable for long-CoT models, so "off-principal" is an interpretive proxy, not a measured Hessian quantity.
- The Three-Gate Theory is a set of mostly one-step, small-step approximations (O(||Δθ||²) error, a Fisher lower bound on a subspace), and the causal link between geometry and bias rests on one model (Qwen3-4B-Base) with two layers scrambled.
- The bf16 argument is partly self-admitted: the authors say sparsity "is amplified by" bf16 but does not vanish entirely, and they note higher-precision storage reduces sparsity without improving task performance, which weakens the claim that sparsity has no functional role.
- The paper's own evidence on spectral preservation and mask avoidance is correlational across checkpoints that differ in data, algorithm and recipe; the claim of invariance to datasets and RL recipes rests on five runs from one base model.
- The PEFT comparisons use short runs (200 steps) and a small number of seeds; the PiSSA "collapse" result is at specific learning rates.
- Nothing is done on decompositions of the weights; the subspace analysis is by SVD of individual matrices, not by any learned component basis.

## Key Quotes
> "for a fixed pretrained model, updates consistently localize to model-preferred parameter regions, remain highly consistent across runs, and are largely invariant to datasets and RL recipes" (Abstract)

> "RL updates exhibit a sub-random overlap with principal weights, indicating a strong tendency to avoid them." (Sec. 4.2)

> "Takeaway 3. RLVR learns off the principals: it preserves spectral geometry, avoids principal weights, and its optimization bias vanishes when pretrained geometry is disrupted." (Sec. 4.4)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint, no comments).
