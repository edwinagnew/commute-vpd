# Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent

## Metadata
- **Authors**: Zhiyuan Li (Princeton), Tianhao Wang (Yale), Jason D. Lee (Princeton), Sanjeev Arora (Princeton)
- **Year**: 2022 (arXiv v1, July 2022)
- **Source**: arXiv preprint (I believe a later NeurIPS 2022 version exists; not verified from the file)
- **DOI/URL**: arXiv:2207.04036

## Summary
The paper characterises when gradient flow on a reparametrised model w = G(x) is exactly mirror flow in the original parameter space w. The key condition is that the reparametrisation is "commuting" (the Lie brackets of the coordinate gradients vanish), and the paper shows this is sufficient, nearly necessary, and that every mirror flow can be realised this way via Nash's embedding theorem. It thereby unifies earlier implicit-bias results for diagonal, quadratic and u^2 - v^2 parametrisations of linear regression.

## Key Findings
- Gradient flow under any commuting parametrisation G is equivalent to mirror flow in w-space with a Legendre function R that depends only on x_init and G, not on the loss (Theorem 4.9). The loss-independence is what lets the implicit bias be stated for any loss.
- Commuting is a checkable condition (Lie brackets [∇G_i, ∇G_j] = 0), and a necessary condition is weaker: Lie-bracket terms must lie in ker(∂G). The authors conjecture this weaker condition is also sufficient (Conjecture 4.11, open).
- Matrix factorisation G(U) = UU^T and G(U,V) = UV^T are non-commuting, and the paper proves gradient flow on UU^T cannot be written as mirror flow for general losses (Corollary 4.12). Non-commuting parametrisations cannot be mirror flows in general.
- Converse: every mirror flow with a Legendre function R is the gradient flow of some commuting parametrisation, obtained by an embedding via Nash's theorem (Theorem 5.1; not read in detail, from Section 1 contributions).
- Recovers and extends implicit-regularisation results for underdetermined linear regression (min-norm/ℓ1-type biases), and proves parameter convergence that earlier works assumed (Theorems 4.14-4.16, Corollary 4.18).

## Methodology
Purely theoretical. Uses differential geometry: vector fields, Lie brackets, flows and Sussmann's orbit theorem, plus convex analysis (Legendre functions, Bregman divergences). The main argument is that under commutativity the gradient-flow solution is x(t) = ψ(x_init; µ(t)) with µ(t) the integrated negative gradient, so Q = conjugate of R is well-defined and the dynamics become mirror flow. Examples are worked by hand (u^2 - v^2, quadratic parametrisations, matrix factorisation). No experiments are reported in the portion read.

## Relevance to Proposal
**Score: 4**

The paper bears on strand 5 (reparametrisation and factorised gradient descent). The proposal's candidate map pulls the gradient step back through W = Σ U_c V_c^T, which is a factorised parametrisation; this paper says such parametrisations are generally non-commuting and so the factorised flow has no mirror-flow description. The word "commuting" here is a technical term about Lie brackets of parametrisation gradients and is unrelated to the project's "commute past" idea, so it is a terminology collision to watch for, not a method to reuse.

## Limitations
- Continuous-time only (gradient flow, infinitesimal step size). Discrete-step behaviour is only covered by later work the paper cites (Ghai et al. 2022), not here.
- Necessary-and-sufficient characterisation is incomplete: the sufficiency of the weaker Lie-algebra condition is only conjectured (Conjecture 4.11).
- The converse construction (Nash embedding) is existence-only and gives no practical or constructive parametrisation.
- Results assume smoothness, regularity (rank-d Jacobian) and a completeness-type Assumption 3.5 on flows, which fails for common polynomial parametrisations such as x^⊙3 (Section 3.2).
- No empirical validation in the parts read. Applicability to neural networks with large D >> d is not addressed.
- I read only the main text through Corollary 4.18 (about the first third of the file). Appendix proofs and Section 5 were not read in detail; claims about them come from the introduction.

## Key Quotes
> "It is shown that gradient flow with any commuting parametrization is equivalent to continuous mirror descent with a related Legendre function." (Abstract)

> "These are both non-commuting quadratic parametrizations." (Example 4.5, Section 4.1, p. 9)

> "Note the necessary condition in (9) is weaker than assuming that G is a commuting parametrization, and we conjecture that it is indeed sufficient." (Section 4.2, p. 11)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv paper).
