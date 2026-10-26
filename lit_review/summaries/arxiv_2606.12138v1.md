# Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders

## Metadata
- **Authors**: Gleb Gerasimov, Timofei Rusalev, Nikita Balagansky, Daniil Laptev, Vadim Kurochkin, Daniil Gavrilov (T-Tech)
- **Year**: 2026
- **Source**: arXiv preprint (2606.12138v1)
- **DOI/URL**: arXiv:2606.12138 (v1)

## Summary
The paper measures per-feature "feature stability" in sparse autoencoders (SAEs): the probability that a feature reappears (cosine similarity at least 0.7 on decoder vectors) in SAEs trained with different seeds. Stable features carry most of the reconstruction and next-token relevance, while unstable features are weak individually, are dominated by surface-form triggers, and yet concentrate in reproducible low-rank decoder subspaces, which the authors argue reflects basis ambiguity rather than noise. A synthetic low-rank model reproduces this split, and pooling deduplicated cross-seed features yields SAEs with comparable explained variance and higher mean stability.

## Key Findings
- Stable and unstable features differ functionally: masking unstable features (4x more of them than stable ones, frequency-matched) barely changes explained variance or next-token loss, while masking far fewer stable features degrades both substantially (GPT-2 layer 7, TopK, 96 seeds).
- Unstable features have effective rank ratio ER/d about 0.59-0.65 versus about 0.80-0.81 for stable sets, and their decoder subspaces are reproducible across seeds (cross-seed SVD explained variance close to within-seed). Logistic-regression classifiers on decoder vectors transfer across seeds.
- In a synthetic dictionary where 20 ground-truth features lie in a shared rank-2 subspace, full-rank features are recovered individually (reappearance near 1) but low-rank ones are not (reappearance about 0.25), mirroring the real-model pattern.
- Instability is structural across settings: a trained-model vs random-model control (Figure 8) shows the stable fraction collapses for random transformers, while SAEBench autointerp scores stay high for both, so autointerp alone does not flag the failure.
- Sparsity mechanism matters: Vanilla ReLU+L1 is most stable but has lower EV; increasing TopK k raises EV and raises the unstable fraction. More training does not remove instability (about 8% unstable plateau at 10B tokens in their setting).
- A feature pool built from most-probable cross-seed features, after brief tuning, recovers near-baseline EV (0.92-ish range in Figure 14) with lower instability; the authors report no stability-EV trade-off in this specific construction.

## Methodology
Train 96 TopK SAEs (plus variants: BatchTopK, JumpReLU, HierarchicalTopK, Vanilla) with different seeds on GPT-2 layer 7 residual activations (dictionary 2^14, k=64, FineWeb, 1B tokens). Feature stability uses an argmax-cosine many-to-one matching rule against an anchor SAE (checked against Hungarian matching, IoU 0.978), with endpoint thresholds epsilon=0.05. Analyses cover activation frequency and magnitude, token entropy, SAEBench autointerp with Qwen3-32B and a GPT-5 blind classification of explanations, frequency-matched masking with EV and CE-under-patching, effective rank and SVD subspace overlap, logistic-regression transfer, a synthetic low-rank dictionary, and a feature-pool construction followed by about 2M tokens of tuning.

## Relevance to Proposal
**Score: 4**

This is a single-model SAE study, not a parameter decomposition, and it does not touch commuting a fine-tuning update past a decomposition. It is relevant to strand 1 (stability, seed dependence and identifiability of decompositions) because it gives a concrete recipe for the question "are individual components identifiable across runs, or only their subspaces?", and its low-rank subspace argument parallels the question of whether a rank-1 subcomponent is a stable unit. The cross-seed matching metric and the trained-vs-random control are reusable ideas for any component-matching check.

## Limitations
- Conditional thresholds: stable/unstable membership depends on theta=0.7 and epsilon=0.05; the authors say the qualitative trends are robust but endpoint sizes are not.
- The subspace result is evidence for one mechanism, not a complete account of instability in real LLM SAEs (the authors say so in the Conclusion).
- The "no stability-EV trade-off" claim is specific to the feature-pool construction and does not hold for single-SAE objectives (Figure 14 shows trade-offs there).
- Main experiments use a single model and layer (GPT-2 L7); other settings are in an appendix table. The synthetic model is small (d=32, r=2, k=8).
- Interpretability results rely on an LLM judge (Qwen3-32B, GPT-5) and SAEBench autointerp, whose reliability the paper itself questions.
- The "Limitations" heading in the PDF text is empty; the limitations appear inside the Conclusion section.
- Cites an "Anonymous Authors (2026)" under-review TMLR submission for a related benchmark, which cannot be checked.

## Key Quotes
> "unstable features are individually non-reproducible but concentrate in reproducible lower-rank subspaces, suggesting that seed dependence often reflects basis ambiguity within a shared region of activation space rather than pure noise." (Abstract)

> "we adopt a many-to-one alternative: for each element from D_1, we find the most similar element from D_2, measured by maximum cosine similarity." (Section 3.2)

> "Our stability metric sharply distinguishes these cases" (Section 7, dead-salmon baseline; Figure 8)

## Comments Summary (if LessWrong/AF post)
Not applicable (arXiv preprint, no comments).
