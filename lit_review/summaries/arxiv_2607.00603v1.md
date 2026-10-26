# Measuring Dead Directions: Decomposing and Classifying Singular Structure off Canonical Alignment

## Metadata
- **Authors**: Tejas Pradeep Shirodkar (IIIT Hyderabad)
- **Year**: 2026
- **Source**: arXiv preprint (single author, not peer-reviewed as far as the document shows)
- **DOI/URL**: https://arxiv.org/abs/2607.00603

## Summary
The paper proposes a descent-free, alignment-free way to read the singular structure of a trained network from one frozen checkpoint. It nominates a "dead direction" (via K-FAC factors or an architecture-specific detector), scans the directional Fisher along it, and reads off the KL order k, the per-direction learning coefficient 1/(2k), and whether the direction is a genuine singularity or a flat gauge. It is a singular learning theory (Watanabe) measurement paper, not a study of fine-tuning updates or parameter decompositions.

## Key Findings
- At a single frozen checkpoint, the log-log slope of the directional Fisher along a nominated dead direction gives the order k (k = 1 + alpha/2). On constructed node-death cells the read returns the planted orders 2, 3, 4 at r^2 = 1.000, and on trained deep linear networks it returns k = 3.00, 4.02, 5.07 for depths L = 3, 4, 5.
- Off-canonical (rotated) dead directions are recovered by constructing the joint mode from K-FAC factors rather than searching the Fisher spectrum; a per-coordinate scan misses them. On a from-scratch ViT (ImageNet-100), squared-ReLU reads k = 3.00 and gelu reads k ≈ 2.0 on rotated subspaces.
- A detector-plus-read pipeline classifies directions into genuine singularities (finite order) and gauges (flat magnitude, e.g., LayerNorm-kernel and attention QK/VO rotations in a fine-tuned DINOv2). The paper reports this as the taxonomy of Table 2.
- The optimizer, not the architecture, sets whether a clean dead structure forms: scaled-polar (orthogonalising) optimizers give axis-aligned node-deaths, vanilla Muon gives diffuse structure with no readable order, and AdamW/RMSProp results depend on target noise.
- The singular fluctuation nu(k) is fixed by the order (nu(2) ≈ 0.173, nu(3) ≈ 0.278, confirmed by sampling on an isolated direction), but a trained network's realised nu falls below it because live parameters absorb part of the data fluctuation (claimed overlap effect, measured on controlled cells).
- The global learning coefficient is assembled from per-direction orders where singular loci are enumerable (matches Aoyagi's closed form on analytic cells); it fails on wide matrix deep linear networks with determinantal loci.

## Methodology
Frozen-checkpoint measurement. K-FAC factors A and G are accumulated over held-out tokens with label-resampled (true) Fisher; a dead direction is nominated from the better-separated factor (an A-G duality), and a cross-layer joint mode is assembled. The directional Fisher u^T F(theta_0 + t u) u is computed by finite-difference JVPs on a 16-point log-spaced grid over t in [1e-2, 0.5], and a single power law is fit over an auto-selected window (highest r^2 with r^2 > 0.95). Experiments use one-block RoPE transformers on modular addition (d=8, d=24), deep linear networks, a teacher-student conv net, a fine-tuned DINOv2 ViT-S/14 on CIFAR-100, and a from-scratch 6-block ViT on ImageNet-100. Comparisons against SGLD local learning coefficient estimates are made on a seven-optimizer modular-addition cohort.

## Relevance to Proposal
**Score: 3**

The paper is about measuring the singular learning coefficient of dead directions at one checkpoint, not about carrying a parameter decomposition (VPD/SPD/APD) through a post-training update. It touches the proposal only at the edges: it shares the "decompose a global complexity scalar into local per-component structure" framing, and its gauge-versus-singularity classification could in principle inform which subcomponents are inert under a gradient step. It does not treat fine-tuning updates, LoRA geometry, or decomposition stability/identifiability, which are the proposal's core strands.

## Limitations
- Single-author preprint with heavy self-citation (the method builds on the author's own companion papers Shirodkar 2026a, 2026b and Shirodkar & Narayanan 2026a, 2026b); the central claims depend on those.
- Authors state the read sees only the fraction of structure its detector family surfaces; the unattributed remainder is not measured, so a global coefficient from named structure is only a lower bound.
- Requires a dead direction to have already formed; the read returns no order during the accumulation phase, which includes standard pretraining. Models studied are small (d=8, d=24 transformers, 6-block ViT), and most are toy or modular-arithmetic tasks.
- Method is validated mostly on constructed cells and small models; the trained-network results for the global coefficient and nu are limited, and the authors list the general determinantal resolver as open.
- The read is per checkpoint; it does not address training dynamics or fine-tuning deltas. The trajectory-based cross-check is only validated under SGD in prior work.
- Several terms are coined or local (e.g., "node-death", "dead-subspace dimension", "purity-matched window") and rely on the companion papers for definitions.
- Appendices (about 500 lines of the 1034-line source) were not read in full; the summary relies on the main text, tables, and Appendix A.

## Key Quotes
> "The read recovers the architecture-predicted order across constructed cells and trained networks, including a fine-tuned vision transformer whose dead structure is the LayerNorm-kernel gauge and a from-scratch one whose compressed MLP forms a node-death at its activation order." (Abstract)

> "The optimizer decides whether a cleanly readable dead structure forms at all, and on this deep transformer the orthogonaliser is what decides it." (Section 5)

> "The hard limit is the first sentence: the read sees the fraction of a real model's singular structure its detector family surfaces (the unattributed remainder is the coverage residual above)." (Section 9)
