# Differentiable Faithfulness Alignment for Cross-Model Circuit Transfer

## Metadata
- **Authors**: Shun Shao, Shay B. Cohen, Binxu Wang, Anna Korhonen, Yonatan Belinkov
- **Year**: 2026 (arXiv 2604.24302; marked "Preprint. Under review.")
- **Source**: arXiv preprint
- **DOI/URL**: https://arxiv.org/abs/2604.24302 (code: https://github.com/jasonshaoshun/dfa-circuits)

## Summary
The paper proposes Differentiable Faithfulness Alignment (DFA), which learns a dense linear map W from node-importance scores of a small source model to node scores of a larger target model, trained with a soft, differentiable faithfulness (KL) objective plus an L1 sparsity penalty. The aim is to predict target-model circuits without running full circuit attribution on the target. On Llama-3 and Qwen-2.5 across six MIB tasks, the strongest transfer is from Llama-3 1B to 3B, with recovery weakening for larger gaps and for Qwen-2.5.

## Key Findings
- On Llama-3 1B→3B with NAP-IG-INPUTS, zero-shot DFA reaches CPR around 0.44 average, close to the direct target-attribution "Gold" reference (0.38 average), and the best per-setting result (0.45) exceeds it (Table 1).
- Controls (random W, scrambled source scores, column-permuted W, fixed depth-based heuristic W) all perform worse than trained DFA on MCQA-style comparisons (Table 2), which the authors read as evidence for a learned, source-specific correspondence.
- Transfer degrades with larger gaps: Llama-3 1B→8B, 3B→8B and all Qwen-2.5 pairs recover only about 28–50% of the Gold MCQA faithfulness (Table 3). Reverse (large→small) transfer is reported as easier.
- Cross-task transfer matrices show structured, partly reusable correspondences; some off-diagonal (cross-task) entries are as strong as the matched-task ones (Figure 4).

## Methodology
- Source circuits: node-level attribution (NAP, NAP-IG-activations, NAP-IG-inputs) on heads and MLP blocks, computed on the source model. Gold references are the same attribution methods run directly on the target.
- Alignment: s_hat_L = s_S W with W in R^{|V_S| x |V_L|}; mask m_L = sigmoid(s_hat_L) interpolates clean and corrupted target activations per node.
- Loss: KL(P_clean || P_intervened) + lambda times the mean mask value (L1). W is trained across many source-score vectors and training tasks.
- Evaluation: CPR (area under the faithfulness-vs-proportion-of-nodes curve, following MIB), across in-distribution, near-distribution (paired arithmetic / ARC tasks) and leave-one-task-out zero-shot settings.

## Relevance to Proposal
**Score: 3**

The paper transfers a structural object (a per-node importance profile) from a small model to a larger one through a learned linear map trained with a faithfulness objective. That is loosely analogous to the proposal's question of whether a simple map Phi(D, dW) can carry a decomposition of W from before an update to after it, and its "does the mechanism persist across scale" finding bears on the proposal's strand 6 (mechanism stability). It does not touch parameter decomposition, fine-tuning updates, training dynamics, or any commuting/reparametrisation step, so its direct use is limited to a possible baseline framing and a cautionary result about where cross-model transfer breaks down.

## Limitations
- Node-level circuits only (heads and MLPs); the authors note edge-level methods are generally stronger, and the cross-model map is evaluated only on attribution-based circuit scores.
- Transfer is inconsistent: it weakens for larger size gaps and is much weaker on Qwen-2.5, so the headline claims rest mainly on one family and one pair (Llama-3 1B→3B).
- The "DFA beats Gold" result is not a clean win. Gold is itself a weak node-level attribution on the target, and the authors acknowledge this without showing the Gold circuits are reliable. CPR values for Gold above 1.0 in Table 3 are not explained, and the metric scale appears inconsistent with Table 1.
- Averages are over six tasks with small differences (e.g. 0.38 vs 0.44 vs 0.45); no variance or seed statistics are given in the main text.
- The alignment matrix W is dense and of size |V_S| x |V_L| (up to about 700 x 1056), so it is specific to a fixed pair of architectures and node indexing; it does not generalise to architectural change or to a model's own updated weights.
- Evaluation is on MIB tasks only, with no fine-tuning or training-dynamics setting, and no comparison to parameter decomposition methods.
- Authors' claim that smaller models "provide cleaner mechanistic signals" is a hypothesis from one comparison, not established.

## Key Quotes
> "The strongest results occur on Llama-3 1B → 3B, where aligned circuits are often competitive with direct node attribution and zero-shot transfer remains effective. Recovery weakens for larger source–target gaps and is substantially lower on Qwen-2.5" (Abstract, p. 1)

> "Rather, circuit localization methods scale poorly with model depth and width (Hanna et al., 2024): attribution can produce cleaner and more faithful circuits on smaller models than on larger ones." (Section 4.2, p. 7)

> "Heuristic W performs poorly across tasks, showing that a simple structural prior based only on relative depth and component type is not enough to explain the gains." (Section 4.3, p. 8)

> "Our study is limited to node-level circuits, where localization is generally weaker than edge-level methods (Mueller et al., 2025)." (Section 5, p. 10)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint, no comments).
