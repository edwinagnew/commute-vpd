# Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression

## Metadata
- **Authors**: Tue M. Cao, Nguyen Do, My T. Thai (University of Florida)
- **Year**: 2026 (arXiv 2605.28567, May 2026; NeurIPS-format submission)
- **Source**: arXiv preprint (NeurIPS checklist present, so submitted to a NeurIPS-style venue)
- **DOI/URL**: arXiv:2605.28567v1

## Summary
The paper proposes representing each sparse autoencoder (SAE) feature as an activation-weighted distribution over the hidden states where it fires (top-K tokens), projecting those states into a shared reference space (the target layer, or a concatenation of all layers), and comparing features with an optimal-transport (Wasserstein) distance. The same distance is used for two tasks: cross-layer feature matching and automatic grouping of feature-circuit nodes into "supernodes". The authors claim theoretical invariance and stability results and report gains over decoder-vector matching baselines, feature-flow baselines, attribution patching, and an LLM selector on GPT-2 and Gemma-2-2B SAEs.

## Key Findings
- Decoder-vector matching (SAE Match, FeatFlow) degrades sharply for non-adjacent layers (e.g. Gemma-2-2B layer 12 to 25: LLM-Eval 1.83 for the proposed method vs 1.26 to 1.23 for baselines). For adjacent layers the methods are close (Table 1).
- On circuit compression (50 nodes to 5 supernodes), the method scores 0.615 (Gemma-2-2B) and 0.681 (GPT-2-small) on an LLM-judged "which node belongs to the caption" task, above FeatFlow, Match, Attrib and an LLM selector (Table 2). Confidence intervals are 90% and overlap with the LLM selector on Gemma-2-2B (0.615 ± 0.015 vs 0.602 ± 0.013).
- Theory: the projected distribution is invariant to positive global rescaling of activations under deterministic TopK (Prop. 1, essentially immediate from ranking invariance); Wasserstein scores are Lipschitz in the input distributions (Thm. 1); matching and supernode assignment are preserved when the empirical perturbation is below half the margin (Thms. 2 and 3).
- Qualitative claim: the method separates "digit addition" features (e.g. "first operand is 5") that baselines confuse, in a blind human ranking study on 100 targets (Figure 2 and 3).
- Modular feature groups: clustering features by co-occurrence in compressed supernodes recovers induction-like and "(" -related groups in incrementing-sequence circuits (Figure 4), reported qualitatively.

## Methodology
- SAEs: open JumpReLU SAEs (Gemma Scope) on Gemma-2-2B MLP outputs, and self-trained TopK SAEs on GPT-2 residual stream.
- Feature representation: top-K token positions by activation; weights normalised over the top-K; projection into a shared space R. For matching, R is the target layer's hidden state at the same tokens. For circuits, R is the concatenation of all layers' hidden states.
- Distance: discrete optimal transport (Wasserstein) between the two weighted token sets with a generic ground cost.
- Evaluation: 100 random feature pairs per setting (LLM-judged similarity via gpt-oss-120b captions, CE and variance-explained when swapping the feature in); 200 circuits for compression, with an LLM picking the supernode member from a caption. Baselines are re-implemented by the authors.
- Theory is in the appendices (Appendix E), with a NeurIPS checklist stating the proofs are complete.

## Relevance to Proposal
**Score: 2**

This paper is about cross-layer SAE feature matching and circuit clustering, not about parameter decompositions (VPD/SPD/APD), low-rank fine-tuning updates, or tracking a decomposition through a gradient step. The only overlap is the broad "interpretability that tracks change across layers or checkpoints" strand (strand 2), and the fact that its distance is defined on activations rather than weights, which is the opposite of the weight-space pullback the proposal wants to test.

## Limitations
- The paper does not release code (NeurIPS checklist, Q5: "We will provide the code once the paper get published"), so the baselines and the LLM judging pipeline are not reproducible from the paper alone.
- Matching and compression are judged by an LLM (gpt-oss-120b) and a single human ranking study. The LLM-judged scores depend on the judge and on captions produced from activation tokens, and the human study size and inter-rater agreement are not visible in the main text I read.
- Evaluation sets are small (100 feature pairs; 200 circuits) and cover two models, both with SAEs trained by the authors or released by others at specific widths.
- The "invariance to rescaling" result (Prop. 1) holds by construction under TopK selection; it is not evidence of semantic quality.
- The margin theorems are conditional on population margins and on estimation error bounds that are not measured empirically in the main text.
- Activation-based representations depend on the chosen top-K and the corpus; the paper does not analyse sensitivity to K or corpus choice in the material I read.
- Caveat on coverage: I read the main body, Tables 1 and 2, and the appendix from about page 29 onward (E.1 to E.3). I did not read the full appendix on the limitations discussion (Appendix A) or the experimental-detail appendix (Appendix C), so claims about those sections are not made here.

## Key Quotes
> "we abandon the standard vector-centric paradigm and instead frame feature comparison as an Optimal Transport (OT) problem" (Section 1, Introduction)

> "Our method performs well on LLM Eval and Matching Accuracy, while maintaining competitive performance on downstream cross entropy loss and variance explained." (Section 5.1)

> "In practice, we choose the reference space R as the space of target layer A, where for each activation token of feature at layer B, we extract the representation at the same token in layer A." (Section 3.2)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint).
