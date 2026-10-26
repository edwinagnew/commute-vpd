# Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning

## Metadata
- **Authors**: Zhiyuan Li, Yuping Luo, Kaifeng Lyu
- **Year**: 2021 (arXiv 2012.09839, v2)
- **Source**: ICLR 2021 (conference paper)
- **DOI/URL**: https://arxiv.org/abs/2012.09839

## Summary
The paper studies the implicit regularization of gradient flow (infinitesimal-step gradient descent) when fitting a symmetric, depth-2 matrix factorization W = UU^T from small initialization. It shows, theoretically and empirically, that this trajectory is equivalent (under stated assumptions) to a simple greedy algorithm, Greedy Low-Rank Learning (GLRL), which adds one rank at a time and only increases the rank when the current rank-constrained solution is not a global minimizer. This argues that rank minimization, rather than nuclear-norm minimization, describes the bias, and it gives a counterexample to the Gunasekar et al. (2017) norm-minimization conjecture.

## Key Findings
- Gradient flow from infinitesimal initialization is, under assumptions (eigenvalue gap, analytic loss, bounded trajectory), equivalent to GLRL: the solution learns rank-1 first, then sequentially adds rank-1 components (Sections 5.1–5.3).
- The first phase is rigorous: the rank-1 limit trajectory is a unique solution determined by the top eigenvector of -grad f(0), with the leading direction set by the largest eigenvalue of the linearized dynamics (Theorem 5.6).
- The nuclear-norm conjecture of Gunasekar et al. (2017) is refuted by a counterexample (Example 5.9), so norm minimization does not fully characterize the bias.
- For depth >= 3, the paper shows GF with small identity initialization follows a deep version of GLRL at least in the early phase; depth weakens the dependence on initialization scale, so rank minimization applies at more practical init scales (Section 6).
- Experiments in Appendix C validate the equivalence; GLRL generalizes better than the related Rank-1 Matrix Pursuit when the ground truth is low-rank (Figure 3).

## Methodology
Mathematical analysis of gradient flow dynamics. The approach linearises the dynamics around the origin (a strict saddle), where the flow behaves like continuous power iteration. The top eigenvector of the Jacobian selects one of two limiting trajectories for infinitesimal initialization, and a perturbation argument then gives the greedy phase structure. Experiments on matrix sensing / full-observation settings check the theory. No language-model or neural-network experiments are reported.

## Relevance to Proposal
**Score: 3**

The paper is a theory result about a toy setting (small-init factorized gradient descent on a symmetric matrix), and it bears only indirectly on the proposal's question of whether a small post-training update δW can be carried by existing parameter-decomposition components. Its relevant idea is that gradient descent builds a weight matrix by greedily adding rank-1 directions in a fixed order, which is a possible mechanistic account of how an update decomposes into components; it also cautions that the correct description of the update may be rank-based rather than norm-based. It does not address pretrained networks, decompositions such as VPD, or fine-tuning.

## Limitations
- Analysis is for symmetric (or reduced-to-symmetric) depth-2 factorization with a convex, C^3-smooth loss; depth >= 3 results cover only the early phase.
- Key results rely on assumptions: a unique positive top eigenvalue of -grad f(0), an eigenvalue gap, locally analytic f, and boundedness of the trajectory (Assumptions 5.5, 5.7; Theorem 5.8).
- Results are for infinitesimal step size and infinitesimal initialization (limits as eta -> 0 and alpha -> 0); finite-step, realistic-init behaviour is only checked empirically.
- Setting is matrix sensing / full-observation toy problems; nothing transfers directly to neural networks with nonlinearities, LoRA, or trained decompositions.
- Authors note that the equivalence is "under some reasonable assumptions"; the generic-ness claims are not fully proven for all losses.

## Key Quotes
> "for depth-2 matrix factorization, gradient flow with infinitesimal initialization is mathematically equivalent to a simple heuristic rank minimization algorithm, Greedy Low-Rank Learning, under some reasonable assumptions." (Abstract)

> "Our theoretical results show that GD performs rank minimization via a greedy process in a broader setting." (Section 1, Our Contributions)

> "This matches with the first warmup example: GD does start learning with rank-1 solutions." (Section 5.2, after Theorem 5.6)

## Comments Summary (if LessWrong/AF post)
Not applicable: this is a peer-reviewed ICLR paper, not a LessWrong/AF post.
