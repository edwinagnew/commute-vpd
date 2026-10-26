# SAEs (usually) Transfer Between Base and Chat Models

## Metadata
- **Authors**: Connor Kissane, Robert Krzyzanowski, Arthur Conmy, Neel Nanda
- **Year**: 2024
- **Source**: LessWrong / Alignment Forum (interim report, posted 2024-07-18; LW score 67)
- **DOI/URL**: https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models

## Summary
The authors train sparse autoencoders (SAEs) on base models and on their chat fine-tunes (Mistral-7B, Qwen 1.5 0.5B, and Gemma v1 2B) and test whether each SAE reconstructs activations from the other model. For Mistral-7B and Qwen they largely do, except for rare high-norm outlier tokens (mostly BOS), and Gemma v1 2B is a clear failure case. They also show that fine-tuning a base Mistral SAE on about 5M chat activations gives a chat SAE comparable to one trained from scratch on 800M tokens.

## Key Findings
- Base-trained SAEs splice into chat models with CE loss recovered close to that of chat-trained SAEs on Mistral-7B (base SAE on chat: 98.85% recovered, chat SAE on chat: 99.01%) and on Qwen 1.5 0.5B. Reconstruction is worse by standard metrics (explained variance 60.9% vs 69.2%), and L0 is higher (126 vs 101).
- Outlier-norm activations (under 1% of tokens, mostly BOS, plus newline and paragraph tokens) are reconstructed poorly across models. The authors filter them out for most evals, and this filtering is a stated caveat.
- SAEs do not transfer on Gemma v1 2B. The authors attribute this to unusually large base-vs-chat weight differences for that model, a result credited to Tom Lieberum.
- Fine-tuning the base Mistral SAE on 5M chat tokens recovers reconstruction quality comparable to a from-scratch chat SAE, without filtering outliers, and gives a sparser SAE (L0 86 vs 101). The fine-tuned SAE's interpretability was not evaluated.
- On instruction-formatted data (Alpaca rollouts, Mistral-7B-Instruct), transfer still holds in CE terms, but both SAEs reconstruct worse than on the Pile.

## Methodology
SAEs were trained with SAELens on a middle-layer residual stream (resid_pre, layer 16 for Mistral, layer 13 for Qwen), on Pile data, following Conerly et al. and Templeton et al. Evaluation used L0, CE loss recovered (zero-ablation baseline), MSE and explained variance, on 50k–100k Pile tokens and a small instruction set. The outlier analysis counts high-norm tokens under a scaled-norm threshold. The fine-tuning experiment used a reduced batch size (256), warmup and decay, and the pretraining L1 coefficient, with no tuning. This is an empirical, reconstruction-based study. No feature-level matching was done.

## Relevance to Proposal
**Score: 6**

This is the closest prior work to the proposal's strand 2 on warm-starting or re-using an interpretability decomposition across a post-training update. Its result that a base dictionary transfers to, and can be cheaply fine-tuned into, a chat dictionary is direct evidence for the "small update, reuse the decomposition" idea, though for SAEs rather than parameter decompositions (VPD). It is weaker on the core question, because it measures reconstruction and never checks whether features are faithful or stable under the update, which the proposal needs.

## Limitations
- Only reconstruction and CE-loss metrics are reported. Feature-level comparison, interpretability of the fine-tuned SAE, and steering/causal checks are all listed as not done.
- Outlier-norm activations are filtered in most evals, which the authors flag as a caveat. The filtering changes what the metrics are blind to.
- Three models, one layer each (mid-residual). The authors note results could differ at other layers and sites.
- Chat SAEs were trained on Pile text, not instruction data, so the chat-side numbers may understate transfer on chat data.
- Gemma v1 2B shows that transfer can fail, and the failure is explained only by a weight-norm observation, not a mechanism.
- The write-up is an interim report, and the authors say the fine-tuning hyperparameters were untuned.
- The fine-tuning comparison is against a single from-scratch SAE, with no seed variance reported.

## Key Quotes
> "SAEs trained on the base model transfer surprisingly well to reconstructing chat activations (and vice versa) on Mistral-7B and Qwen 1.5 0.5B." (Executive Summary)

> "We think the fact that SAEs can transfer between base and chat models is suggestive evidence that: 1. The residual streams of the base and chat models are often extremely similar; 2. Base-model SAEs can likely be applied to interpret and steer chat models, without the need to train a new chat SAE from scratch." (Investigating SAE Transfer between base and chat models)

> "We find that the difference in weights between Gemma v1 2B base vs chat is unusually large compared to other fine-tuned models, explaining this phenomenon." (Introduction)

> "We fine-tune our base Mistral-7B SAE (on 5 million chat activations) to cheaply obtain an SAE with competitive sparsity and reconstruction fidelity to a chat SAE trained from scratch (on 800M tokens)." (Executive Summary)

## Comments Summary (if LessWrong/AF post)
The provided file contains only the post body. It has no comment thread, so no commenter pushback or extensions are recorded here. The post's Limitations section notes that a comment by Leo Gao motivated the authors' point that zero-ablation CE is an unrealistic baseline. The post also says that related upcoming work by some of the authors reaches similar conclusions while addressing these limitations.
