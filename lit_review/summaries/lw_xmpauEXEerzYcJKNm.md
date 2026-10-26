# What We Learned Trying to Diff Base and Chat Models (And Why It Matters)

## Metadata
- **Authors**: Clément Dumas (post author; the post thanks Cameron Holmes and Bart Bussman; work done as part of the MATS 7 extension)
- **Year**: 2025
- **Source**: LessWrong (posted 2025-06-30, score 106)
- **DOI/URL**: https://www.lesswrong.com/posts/xmpauEXEerzYcJKNm/what-we-learned-trying-to-diff-base-and-chat-models-and-why

## Summary
The post argues that model diffing (studying what fine-tuning changes internally, here base vs. chat Gemma-2 2B) is a promising direction, and reports that sparse-dictionary methods have problems in this setting. Crosscoders produce many spurious "chat-only" latents that are artifacts of the L1 penalty; switching to BatchTopK removes the artifacts, and training SAEs directly on `chat - base` activations ("diff-SAEs") recovers more of the behavioural difference than crosscoders. The authors also say the field needs behavioural evaluation (not just reconstruction metrics) and point to a planned diffing toolkit and model-organism benchmark.

## Key Findings
- Most "chat-only" latents from a standard L1 crosscoder (by representation norm) are not truly chat-specific. Latent Scaling (the authors' metric, a per-model regression coefficient β) shows they are present in both models at different intensities. The authors attribute this to complete shrinkage and latent decoupling caused by the L1 penalty.
- BatchTopK crosscoders remove the spurious chat-only latents and find more chat-specific latents that are judged interpretable (refusal triggers, false-information detection, questions about the model's own experiences).
- Diff-SAEs (trained on `chat - base`) and chat-SAEs find more chat-specific latents than the crosscoder, without the frequency artifacts the crosscoder shows.
- On a KL-recovery metric (patching latents into base activations and measuring how closely the output matches the chat model), the crosscoder recovers about 60% of the KL gap, while the diff-SAE recovers about 88%. Diff-SAEs have lower fraction of variance explained (82% vs. 91%), which the authors read as behaviourally irrelevant noise being ignored.
- Applying Latent Scaling to a base-model SAE (in the style of OpenAI's emergent-misalignment diffing) did not give interpretable chat-specific latents in this setting.
- Model stitching (replacing base layers with chat layers) gives a smooth interpolation of behaviour; identity answers switch from "ChatGPT" to "Gemma" only after layers 20-25 (an anecdotal observation in a footnote).

## Methodology
- Crosscoders (Anthropic-style, middle layer of Gemma-2 2B base and chat), trained with either L1 or BatchTopK.
- Latent Scaling: for each latent j with decoder direction d_j, fit a scalar β_j^m per model m by least squares on that model's activations; specificity is ν_j = β_base / β_fine-tuned (closed form in Appendix B). ν ≤ 0.3 is treated as chat-only.
- Chat-SAE (layer 13 of the chat model) and diff-SAE (on `chat - base`), with BatchTopK, k=100.
- Behavioural evaluation: add latent reconstructions (or errors) to base activations, pass through the chat model's remaining layers, and measure KL to the true chat output. Baseline is base activations run through the chat model.
- Qualitative: latent inspection, steering of "How do I make cheese?", and a stitching experiment. No ground-truth diff was available, so validation relies on interpretability and KL.

## Relevance to Proposal
**Score: 4**

This is about SAE and crosscoder model diffing (strand 2 of the proposal, base-to-chat feature differences), not about parameter decompositions or commuting updates into the weights. Its most relevant points are the warning that a shared dictionary for base and fine-tuned models can produce artifact "model-specific" features, the use of behavioural KL recovery as the criterion for a diff method, and the explicit suggestion that parameter decomposition of the weight difference is an open direction. It does not test whether a weight update can be mapped onto an existing decomposition.

## Limitations
- One model pair (Gemma-2 2B base vs. chat) and one layer/dictionary setting; generalisation is not tested.
- No ground truth for what the diff is, so "interpretable" and "chat-specific" are judged by inspection and by the authors' own metrics. The post itself says this is the central difficulty.
- Several explanations are speculative and not tested: the causes of the base-SAE failure (listed as possibilities in Appendix A), and the claim that "any activation function that optimises L0" would work similarly (footnote 3).
- The "root cause" language (L1 penalises activation strength) is the authors' interpretation; the evidence is that BatchTopK removes the artifacts, not a controlled ablation of the mechanism.
- The controlled comparison of diff-SAE vs. chat-SAE (expansion factor 1, same k) is only partial; the KL results come from a single evaluation pipeline and token distribution.
- The post reports results as a progress update ("our next steps"). The diffing-toolkit and evaluation framework were not yet released at the time of writing.

## Key Quotes
> "The key insight is that if a concept only exists in the fine-tuned model, its base model representation should have zero norm (since the base model never needs to reconstruct that concept)." (Background)

> "The root cause is that L1 penalties optimize for total activation strength rather than true sparsity." (The Fix: BatchTopK Crosscoders)

> "Crosscoders leave significant behavioral differences on the table: When we use the crosscoder's full reconstruction of the difference, we only recover 60% of the KL gap." (diff-SAEs Excel at Capturing Behavioral Differences)

> "Our results show that it's unclear if crosscoders should be the default sparse dictionary technique for model diffing." (Conclusion)

> "Could we skip activations entirely and decompose weight differences directly? ... Alternatively, Parameter Decomposition of differences might also be promising." (Conclusion, Weight-based Analysis)

## Comments Summary (if LessWrong/AF post)
The post has one comment thread (Pawan, score 2; author reply, score 2).
- Pawan asked whether applying the diff-SAE idea to crosscoders would fix the shared-latent bias.
- The author replied that this has been considered but not tested. The suggested trick is to add a term L_diff to the crosscoder loss, giving L = MSE(e_chat) + MSE(e_base) + λ·2 e_chat·e_base with λ in [-1, 0]. At λ = -1 the objective focuses on reconstructing the difference (close to a diff-SAE, except the input is chat and base rather than their difference), and at λ = 0 it is the standard crosscoder objective.
- The author plans to test the downstream usefulness of this modification and Matryoshka loss within the diffing toolkit. No results are given in the thread.
