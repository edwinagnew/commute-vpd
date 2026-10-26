# The Loss Does Not See the Basis, but Adam Does

## Metadata
- **Authors**: Devender Singh (Memorial University of Newfoundland)
- **Year**: 2026
- **Source**: arXiv preprint (2608.05136)
- **DOI/URL**: https://arxiv.org/abs/2608.05136 (code: https://github.com/idevender/loss-basis-adam)

## Summary
The paper argues that for a factored model W = UV^T, the loss is invariant to a gauge rotation (U, V) -> (UQ, VQ), and that an optimizer reproduces gradient flow's low-rank implicit bias only if its update is gauge-equivariant. Gradient descent, momentum, shared-scalar Adam, Muon and Shampoo are equivariant; Adam, RMSProp, signSGD, Lion and Adafactor are not, and they select higher-rank interpolants with worse ground-truth recovery in matrix sensing, in attention-head twin experiments, and on two hyperspectral datasets at matched training loss.

## Key Findings
- Equivariant memoryless update rules have an exact structure: Delta = H(GG^T) G, i.e. a left preconditioner determined by the Gram matrix of the gradient (Theorem 4.5). Any fixed entrywise nonlinearity at zero optimizer state (Adam, RMSProp, signSGD, Lion) breaks equivariance (Proposition 4.3).
- On 40x40 rank-3 matrix sensing driven to interpolation, equivariant methods reach recovery error 0.00-0.29 while coordinate-wise methods reach 0.42-0.57 (Table 2). A one-parameter dial from Adam (p=1) to shared-scalar Adam (p=0) improves recovery monotonically (0.570 -> 0.201) and lowers effective rank (14.5 -> 5.4) (Section 7).
- In a 2-layer transformer with gauge-equivalent initializations, Adam's twins split by about 3.6e-3 in logit distance at step 1 (float-precision for equivariant methods), and the per-head invariant W_Q^T W_K ends up 56% apart in relative Frobenius distance (Section 6, Table 3). The paper reports this as a matter of basis dependence, not numerical error, with a noise-twin control.
- A spectral-tail phase diagram (10 seeds) finds that within the equivariant class, Muon's equal-rate schedule is exact for exactly low-rank targets but cedes to gradient descent at roughly 4% tail energy (Section 8, Table 4).
- On Indian Pines and Pavia University (rank-48 factored completion, matched training loss), GD gives 43-44% lower held-out RMSE than Adam at the lower sampling density, with lower effective rank; the gap narrows to 23-28% at the higher density (Section 9, Table 5).

## Methodology
Theory (gauge group O(k), equivariance definition, first-step defect for Adam, structure and transfer theorems, all proved in appendices) combined with controlled experiments: nine optimizers on underdetermined matrix sensing run to interpolation (loss below 1e-7), with learning-rate sweeps, cosine-decay controls, and an H100 replication ladder (n = 64 to 256, 10 seeds). Attention-head "twin" experiments train two gauge-equivalent copies of a small transformer on modular addition and a character-level LM, with an A = I twin and a 1e-7 noise twin as controls. Real-data tests use hyperspectral matrix completion with train-only learning-rate selection and matched-training-loss comparison.

## Relevance to Proposal
**Score: 3**

The paper does not study parameter decomposition (VPD/SPD/APD), post-training updates, or interpretability; its relevance is indirect. It bears on the proposal's "gradient pulled back through W = sum U_c V_c^T" idea (strand 5): it says the factored-gradient dynamics are basis-free only for equivariant optimizers, so a commute map that is defined through the subcomponent basis could depend on whether the post-training update was computed with Adam (which almost all fine-tuning uses). It also gives a concrete test (gauge twins, noise twins) for whether a proposed map is basis-independent, and it states that LoRA's fixed rank "breaks the mechanism", which is a caveat for LoRA-based deltas.

## Limitations
- Synthetic matrix sensing and two hyperspectral scenes; no LLM, no parameter decomposition, no fine-tuning experiment. Transformer evidence is a 2-layer and 4-6 layer model on toy tasks.
- Mechanism claims rest on interpolation-regime results at a fixed computational budget; the paper itself says the two-class split is "a statement about this computational budget and this size range", with two flagged exceptions (untuned Shampoo, long-anneal signum).
- The paper names Muon's transformer divergence as numerical chaos, not basis dependence, which weakens the equivariance-as-explanation reading for that optimizer.
- Hyperspectral results depend on the learning-rate selection rule (the paper reports GD advantage 13% under per-method best-rate selection vs 43% under train-only selection).
- The merging consequence (Section 6) is explicitly untested end to end.
- The main theorems apply to the memoryless, common-scalar flow; scalar-Adam with momentum is only empirically tied to gradient flow (Remark 4.7).
- Several authors' key claims are on the author's own prior method (FlowAdam); the paper's own sign-descent counterexample (Appendix D.10) lacks a mechanistic account.

## Key Quotes
> "Gradient flow's low-rank mechanism is available to an optimizer only if that optimizer is gauge-equivariant, a condition necessary for the transfer but not sufficient for low-rank recovery." (Abstract)

> "Basis choice is therefore not a tuning detail but a decision about which interpolant the optimizer selects." (Abstract)

> "In short, what Adam learns in an attention head depends on an arbitrary choice of coordinates that has no effect on the model's input–output behavior." (Section 6)

> "By fixing the rank rather than learning it, LoRA breaks the mechanism discussed in this paper; it may, however, still find better solutions inside the lower-dimensional subspace if they exist." (Section 10.1)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint; no comment thread).
