# Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning

## Metadata
- **Authors**: Dan Braun, Jordan Taylor, Nicholas Goldowsky-Dill, Lee Sharkey
- **Year**: 2024
- **Source**: arXiv preprint (Apollo Research / MATS); v2
- **DOI/URL**: https://arxiv.org/abs/2405.12241 (code: https://github.com/ApolloResearch/e2e_sae)

## Summary
The paper proposes training sparse autoencoders (SAEs) end-to-end: instead of minimising reconstruction MSE at one layer, the SAE output replaces the activations and the loss is the KL divergence between the original and patched model's output logits (optionally plus MSE reconstruction at downstream layers). On GPT2-small and TinyStories-1M, these "e2e" SAEs explain more of the model's cross-entropy with fewer features per datapoint and fewer alive features overall than standard local SAEs, with comparable automated interpretability scores. The trade-off is a 2-3.5x longer training time, lower reconstruction fidelity at downstream layers (for plain e2e), and features that vary across random seeds.

## Key Findings
- At matched CE-loss increase, the local SAE (SAElocal) needs more than twice the active features per datapoint (L0) of SAEe2e and SAEe2e+ds, and needs more alive dictionary elements overall (Section 3.1, Table 1; CE-loss increase below 45% of local's for a given L0).
- Plain SAEe2e has much worse reconstruction error of the original activations at every later layer, i.e. activations take a different computational path, while SAEe2e+ds (adding downstream MSE) recovers reconstruction close to SAElocal while keeping the efficiency gains (Section 3.2, Figure 2).
- SAEe2e features are not stable across random seeds; SAElocal and SAEe2e+ds features are (Section 3.3.2). Many e2e features do not align with local SAE features (Section 3.3.3).
- Local SAEs show more feature splitting (higher within-SAE nearest-neighbour cosine similarity) than e2e SAEs (Section 3.3.1), but this is a geometric proxy only.
- Automated interpretability scores are equal or better for SAEe2e+ds vs SAElocal (significant at layers 2 and 6 for matched CE loss; no difference at matched L0), supporting the claim that efficiency does not cost interpretability (Section 3.4, Table 4). Shrinkage (feature suppression) is strongest for plain e2e SAEs (Appendix B.1, Table 5).

## Methodology
Empirical. The SAE is an encoder (affine + ReLU) and a unit-norm dictionary, trained with L1 sparsity plus one of three losses: local MSE, e2e KL divergence with the base model frozen, or e2e KL plus downstream MSE (weighted by beta, split across downstream layers). Training uses OpenWebText (400k samples, context 1024), Adam, learning rate 5e-4, and a sweep over the sparsity coefficient to trace L0 vs CE-loss-increase Pareto frontiers. Metrics: CE-loss increase, L0, alive dictionary elements, downstream reconstruction MSE, cosine-similarity geometry across seeds and types, and automated interpretability (Bills et al. style, GPT-4-turbo explainer, GPT-3.5 simulator) on about 200 features per SAE. Qualitative feature inspection and Neuronpedia hosting support the interpretability claims.

## Relevance to Proposal
**Score: 5**

The paper is relevant to strand 2 of the proposal (interpretability that tracks what the network computes, and training objectives that target functional rather than reconstruction importance) and to the "does the new run stay on the same computational pathway" question, since the downstream-reconstruction term in e2e+ds is an explicit pathway-faithfulness constraint akin to what the commuting step needs. It is not about parameter decomposition, fine-tuning updates, or weight-space maps, so it does not address the core question of Phi(D, dW) directly. Its seed-instability finding for e2e features is a useful analogue for the "brittleness vs. structure" branch of the proposal.

## Limitations
- Only SAEs on residual-stream activations; no parameter decomposition, weight-space method, or fine-tuning updates are studied.
- Detailed analysis concentrates on one layer (layer 6 of GPT2-small); other layers and TinyStories runs are mostly summarised in appendices, and some analyses (e.g. geometry, Tinystories) were not run for all configurations.
- Interpretability is measured with an automated proxy (Bills et al. style) and small qualitative samples; the authors acknowledge it is an "imperfect metric".
- Seed-robustness is shown for a few runs only (two seeds in key comparisons).
- E2e training costs 2-3.5x wall-clock time; the authors compare only against local SAEs trained with more compute (larger dictionary or more data), not against other functional-importance methods.
- The downstream-reconstruction loss weight and KL scaling were tuned by hand with limited exploration (Appendix D), and the authors note alternative weightings are likely better.
- The "functional importance" definition is operational (explains behaviour on the training distribution), not a ground-truth feature criterion.
- Plain e2e SAEs have large downstream reconstruction error and feature suppression, which the authors flag as a risk of unintended computational pathways.

## Key Quotes
> "SAEs may learn more about the structure of the datatset than the computational structure of the network. There is therefore only indirect reason to believe that the directions found in these dictionaries are functionally important to the network." (Abstract)

> "Even though SAEe2es explain more performance per feature than SAElocals, they have much worse reconstruction error of the original activations at each subsequent layer (Figure 2). This indicates that the activations following the insertion of SAEe2e take a different path through the network than in the original model." (Section 3.2)

> "features learned by SAEe2e are quite different for different seeds. This suggests there are many different sets of SAEe2e features that achieve the same output distribution, despite taking different paths through the network." (Section 3.3.2)

> "Note, however, the inclusion of the intermediate reconstruction terms means that L e2e+downstream may encourage the SAE to learn features that are less functionally important." (Section 2.4)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
