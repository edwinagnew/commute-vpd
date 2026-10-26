# Aligned Training: A Parameter-Free Method to Improve Feature Quality and Stability of Sparse Autoencoders (SAE)

## Metadata
- **Authors**: Michał Brzozowski, Neo Christopher Chung (Samsung AI Center Warsaw; University of Warsaw)
- **Year**: 2026 (arXiv 2605 identifier)
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2605.18629

## Summary
The paper proposes "aligned training," a reparameterization of sparse autoencoders (SAEs) in which each feature's encoder row is projected onto the affine hyperplane where its inner product with the feature's decoder column equals one. The authors argue this removes a degeneracy in SAE training, reducing dead features, improving reconstruction, and increasing feature consistency across random seeds, with no extra hyperparameters or losses. The evaluation is limited to SAEs on Pythia 160M and Gemma 2 2B using the SAEBench protocol.

## Key Findings
- The per-feature "alignment score" (encoder row dotted with decoder column) is reported to be bimodal across the SAEs tested, and correlates with MCS (Pearson r = 0.65) and autointerpretability (r = 0.32). These correlations are the authors' empirical observations on a small set of SAEs, not an established result.
- Enforcing alignment = 1 by construction (Section 3.2) is reported to give reconstruction Pareto improvements over standard ReLU SAEs at matched L0 across two models, three dictionary sizes (4K, 16K, 65K) and four sparsity penalties; it also extends to TopK and BatchTopK (mainly in the low-sparsity regime).
- Standard ReLU SAEs at dictionary size 4K show roughly 20% dead features; aligned training is reported to reduce dead features to near zero without resampling or auxiliary losses.
- Mean min cosine similarity (MMCS) across seeds for TopK SAEs (Gemma 2 2B, layer 12, 65K, k=20) is higher for aligned training throughout training. The authors argue the gap opens early enough to show it is not an initialization artifact.
- Combined with p-annealing (Pythia 160M, 4K), aligned training nearly eliminates dead features where p-annealing alone plateaus at about 73% alive, but p-annealing has marginal reconstruction advantage at the highest sparsity.
- On the SCR benchmark, aligned training wins on Pythia across sparsities; on Gemma it is marginally behind at very low sparsity (L0 <= 110) and ahead at higher L0.
- Weight tying (a stronger constraint) reduces dead features but hurts reconstruction, which the authors use to argue the milder hyperplane constraint is better.

## Methodology
Reparameterization: an unconstrained matrix A is projected per feature onto {v : v · W_dec,i = 1}, giving the encoder row W_enc,i = A_i + (1 - A_i · W_dec,i) W_dec,i / ||W_dec,i||^2. Biases and the decoder are trained by ordinary gradient descent. The motivation is a toy model (n=2, m=1, no sparsity) where perfect reconstruction forces the alignment score to 1. Training uses 50M tokens from The Pile, evaluated on OpenWebText, with the SAEBench protocol and Adam (lr 3e-4), on one H100 (about 13 minutes per run). A 500M-token scaling check is reported in Appendix H. Metrics are explained variance, recovered CE loss, dead-feature fraction, MMCS across seeds, and SCR. The paper is almost entirely empirical with small theory content.

## Relevance to Proposal
**Score: 3**

This is an SAE-training paper, not a parameter-decomposition paper, so it does not address the core question of carrying a post-training update through a decomposition Φ(D, δW). Its one tangential link is the coupling of encoder and decoder through a per-feature constraint, which loosely parallels the factorised U_c V_c^T structure the proposal would pull gradients through, and its seed-stability metric (MMCS) is a comparable identifiability measure for strand 1.

## Limitations
- Authors state the evaluation covers only two model families (Pythia 160M, Gemma 2 2B) and a fixed set of SAE architectures; scaling to larger models is open.
- The "bimodal across all modern architectures" claim rests on three pretrained SAEs (Appendix C, E) and is stated more broadly than the evidence supports.
- Stability improvement is shown by MMCS on TopK only in the main figure; no statistical treatment of seed variance is reported in the text.
- The stability claim is argued against initialization artifacts, a failure mode the same authors describe in concurrent work (Brzozowski and Chung, arXiv 2606.02061) for archetypal SAEs; the paper does not rule out a similar confound here in its own comparison.
- The paper mixes optimization claims with mechanistic explanations (e.g., "rotational degrees of freedom") that are plausible but not directly tested.
- Interpretability gains are measured with proxies (MCS correlation, autointerp correlation, SCR); the SCR result is mixed on Gemma.
- No downstream causal or steering evaluation of the aligned features is reported.
- Authors note the p-annealing combination lacks a reconstruction advantage at the highest sparsity levels.

## Key Quotes
> "Aligned training enforces a geometric constraint between the encoder and decoder such that their inner product equals one for every feature, which removes a source of degeneracy in the SAE training without adding any hyperparameters." (Abstract)

> "Since x and w_dec must be parallel, writing x = α w_dec and substituting yields w_enc · w_dec = 1: perfect reconstruction forces the alignment score to one." (Section 3.1, toy model)

> "Our empirical evaluation focuses on two model families (Pythia 160M and Gemma 2 2B) and a fixed set of SAE architectures." (Section 5, Limitations)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint; no comment thread).
