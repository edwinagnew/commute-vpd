# LoRA vs Full Fine-tuning: An Illusion of Equivalence

## Metadata
- **Authors**: Reece Shuttleworth, Jacob Andreas, Antonio Torralba, Pratyusha Sharma (MIT CSAIL)
- **Year**: 2025 (arXiv 2024, v3; published at NeurIPS 2025)
- **Source**: NeurIPS 2025 (39th Conference on Neural Information Processing Systems)
- **DOI/URL**: arXiv:2410.21228 (https://arxiv.org/abs/2410.21228)

## Summary
The paper argues that LoRA and full fine-tuning, although they reach similar task accuracy, produce weight updates with different spectral structure. LoRA-tuned matrices acquire high-ranking singular vectors with near-zero cosine similarity to any pretrained singular vector ("intruder dimensions"), and full fine-tuning does not. The authors causally link these intruder dimensions to forgetting of pretraining behaviour by rescaling them after training.

## Key Findings
- LoRA introduces intruder dimensions (top-ranked singular vectors dissimilar to all pretrained singular vectors) in LLaMA2-7B and RoBERTa-base. Full fine-tuning almost never produces them in the top 10 singular vectors, across the epsilon values tested. Intruder counts are robust across epsilon and k and across LoRA variants (AdaLoRA, LoRA+, PiSSA, VeRA), although the variant tests are described as preliminary.
- At matched task accuracy, LoRA forgets less pretraining distribution than full fine-tuning (LLaMA2-7B and RoBERTa-base). This is not guaranteed: at high learning rates or with alpha = 8, LoRA can forget more. Models with more intruder dimensions forget more (Spearman rho = 0.971 across RoBERTa training runs; rho = 0.59 for LLaMA2-7B across epochs). No significant correlation with test accuracy was found.
- Scaling down the top intruder dimension (lambda = 0.7 or 0.9, sometimes 0.3 for LLaMA2-7B r=256) reduces pretraining loss (forgetting) with little or no drop in target-task accuracy. Scaling up increases forgetting. Scaling a non-intruder singular vector of similar magnitude does not show this pattern.
- In sequential (continual) fine-tuning of RoBERTa over six tasks, LoRA degrades faster than full fine-tuning, and intruder dimensions accumulate with each task. Full fine-tuning keeps the pretrained spectral structure.
- Intruder dimensions grow in singular value and change direction over training. Their count rises with fine-tuning dataset size up to a saturation point, and they disappear as LoRA rank approaches full rank (r = 2048 on LLaMA2-7B math shows none).

## Methodology
- Spectral analysis: SVD of pretrained and fine-tuned weight matrices. For each of the top-k fine-tuned singular vectors, compute maximum absolute cosine similarity with all pretrained singular vectors. A vector below threshold epsilon is an intruder dimension (Algorithm 1; k = 10 by default). Counts are summed across matrices.
- Models: LLaMA2-7B fine-tuned on math and code (checkpoints from Biderman et al. 2024), LLaMA-7B on Alpaca, and RoBERTa-base fine-tuned on six classification tasks.
- Forgetting: average of HellaSwag, WinoGrande and ARC-Challenge for LLaMA2-7B; pseudo-log-likelihood on pretraining data for RoBERTa.
- Causal test: rescale the top intruder dimension in each weight matrix to W = W0 + dW + (lambda - 1) u_i sigma_i v_i^T, sweep lambda, and measure accuracy and pretraining loss. The control is the neighbouring singular vector with similar magnitude.
- Continual learning: RoBERTa trained sequentially on MNLI, QQP, SST-2, SIQA, WinoGrande and FEVER, merging LoRA into the weights after each task.

## Relevance to Proposal
**Score: 6**

The paper's main contribution is a spectral comparison of fine-tuning updates (dW) against the pretrained SVD basis, which falls under strand 3 of the proposal (spectrum and alignment of dW with pretrained singular vectors, LoRA geometry). Its causal intervention on individual singular directions is also a useful precedent for testing whether a small update, decomposed into components, carries the behavioural change. It does not address parameter decompositions (VPD, SPD, APD), subcomponent projection, or the pullback map Phi, so it supports the premise that post-training updates have identifiable structure rather than testing the commuting map itself.

## Limitations
- The authors state that intruder dimensions are "an empirical observation of LoRA". Their causes (learning rate, alpha, rank) are only suggested by experiments, not established.
- The detection rule depends on the hyperparameters epsilon and k. The paper reports robustness over ranges, but the threshold is still a user choice.
- The causal test rescales only the single top intruder dimension per matrix. The correlational results across epochs and runs are weaker (rho = 0.59 for LLaMA2-7B).
- Models are limited to LLaMA-7B-scale decoder models and RoBERTa-base. Continual learning uses only RoBERTa with classification tasks. Variants of LoRA are only preliminarily tested.
- Forgetting is measured with a small set of benchmarks (HellaSwag, WinoGrande, ARC) or pseudo-loss on a pretraining sample. This is a narrow proxy for general capability.
- Full fine-tuning is compared under its own hyperparameters. Results may depend on that choice. I did not read the full appendices (lines beyond the main text and references), so appendix-level caveats may be missing.
- Intruder dimensions are defined relative to the pretrained SVD basis. The paper does not study whether they would be detected under a different basis or for non-attention matrices specifically.

## Key Quotes
> "weight matrices trained with LoRA have new, high-ranking singular vectors, which we call _intruder dimensions_, while those trained with full fine-tuning do not." (Abstract)

> "This shows that while LoRA in general does forget less than full fine-tuning, it is not a guarantee." (Section 4, "An increase in intruder dimensions leads to an increase in forgetting")

> "Across the board, scaling down intruder dimensions seems to have little impact on test accuracy but a major impact on forgetting." (Section 5, "Scaling down intruder dimensions reduces forgetting")

> "It is important to note that intruder dimensions are an empirical observation of LoRA." (Section 3, "Experimental and theoretical justification for why intruder dimensions occur")

## Comments Summary (if LessWrong/AF post)
Not applicable. This is an arXiv/NeurIPS paper.
