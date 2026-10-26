# Stability of Low-Rank Implicit Regularization in Perturbed Deep Matrix Factorization

## Metadata
- **Authors**: Jingzhe Wang (University of Pittsburgh, Informatics and Networked Systems); Hung-Hsu Chou (University of Pittsburgh, Mathematics)
- **Year**: 2026 (inferred from arXiv ID 2605; no date printed in the source text)
- **Source**: arXiv preprint (v2), mathematics / optimization theory
- **DOI/URL**: arXiv:2605.28613v2

## Summary
The paper studies gradient descent on deep matrix factorization (a product of N factors fitted to a symmetric target) and asks when the "low-rank phase" of the trajectory, where the iterate's effective rank sits near that of the best rank-L approximation of the target, is observable and survives additive perturbation of the target. It gives sufficient spectral conditions for a nonempty low-rank interval in the noiseless case, then eigenvalue-level convergence and perturbation-stability bounds for the noisy case. Numerical illustrations are on a synthetic n = 20 problem.

## Key Findings
- Noiseless case (N = 2, identity initialisation alpha·I): sufficient conditions on the spectrum, initialisation scale and step size guarantee a nonempty low-rank window [T_0, T_1]. The authors reduce the existence condition to a gap between lambda_L and lambda_{L+1} plus a small enough step size (Theorem 3.6).
- Under additive symmetric perturbation E, eigenvalues of the iterate converge to the (nonnegative part of the) perturbed eigenvalues, with iteration complexity that degrades with ||E|| (Sections 4.2–4.3).
- Stability: over a perturbed interval, the effective rank of the iterate stays close to that of the rank-L approximation of the noiseless target, with an error term that grows with ||E|| (Theorem 5.1); the interval endpoints shift by an explicit amount in ||E|| (Theorem 5.2).
- The Frobenius approximation error of the iterate to the best rank-L approximation is also bounded in terms of ||E|| (Theorem 5.3).
- Numerically, the plateaus disappear when the step size is raised past the stated bound (eta = 0.1 vs. the bound ~0.009 in the example), and larger noise shortens the rank-2 plateau (Figures 1–2).

## Methodology
Purely analytical. The authors build on the non-asymptotic gradient-descent analysis of Chou et al. (2024, ACHA) for deep matrix factorization, track eigenvalues of the iterate along the trajectory (the iterate is diagonal in the target's eigenbasis under identical identity initialisation, Lemma 4.1), and combine these dynamics with Weyl's inequality, Davis–Kahan sin-theta bounds and a sub-Gaussian concentration bound for the noise. Numerical checks use a single synthetic 20×20 target with a few chosen spectra.

## Relevance to Proposal
**Score: 2**

The paper is a matrix-analysis result about where gradient descent on a factorised symmetric matrix plateaus at low rank. It touches the proposal only tangentially: it gives a formal sense in which a low-rank "phase" of a factorised update survives a perturbation of the target, which is loosely analogous to asking whether a decomposition of W stays valid after a small update δW. It does not address parameter decompositions of neural networks, interpretability, fine-tuning updates, LoRA or SAE/transcoder methods, or any model with nonlinearities.

## Limitations
- Analysis is restricted to symmetric PSD targets, N = 2 in the simplified formulas, and identity initialisation; the authors themselves note nonsymmetric and indefinite cases are future work.
- Several results rely on stated spectral gap conditions (e.g. Lemma 3.1 and Lemma 3.5 monotonicity assumptions) that the authors say are "non-trivial to establish" and only partly verified in the text; the constants (e.g. K_epsilon ≈ 4.2 in Section 6) are not deeply explored.
- Probabilistic bounds for the noise are generic sub-Gaussian bounds, not sharp for any concrete perturbation model (acknowledged by the authors).
- Empirical support is one small synthetic example; no real models or training data.
- Deep linear networks only; the authors state extension to nonlinear models is open.

## Key Quotes
> "We then analyze the perturbed problem, where the target matrix is subject to an additive perturbation. By studying the perturbed gradient descent dynamics at the eigenvalue level, we prove convergence guarantees and quantify how the perturbation size affects iteration complexity and eigenvalue recovery." (Abstract, p. 1)

> "Finally, we establish stability of the low-rank phase under perturbation: the effective rank of the iterates remains close to that of the rank-L approximation of the noiseless target over a perturbed low-rank interval, with explicit dependence on the perturbation size." (Abstract, p. 1)

> "Future directions include extending the analysis to nonsymmetric or indefinite settings, deriving sharper probabilistic bounds for concrete random perturbation models, and developing analogous stability results for broader classes of nonlinear models." (Section 7, p. 31–32)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint, no comments).
