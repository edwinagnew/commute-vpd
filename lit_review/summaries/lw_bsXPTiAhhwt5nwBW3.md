# Do Sparse Autoencoders (SAEs) transfer across base and finetuned language models?

## Metadata
- **Authors**: Taras Kutsyk (sole author listed; code repo credited to tommasomncttn)
- **Year**: 2024
- **Source**: LessWrong (project submission for the AI Safety Fundamentals course, BlueDot Impact); score 28 at time of capture
- **DOI/URL**: https://www.lesswrong.com/posts/bsXPTiAhhwt5nwBW3/do-sparse-autoencoders-saes-transfer-across-base-and

## Summary
The post asks whether an SAE trained on a base language model still works on a finetuned version of that model, without retraining the SAE. It compares two base/finetune pairs (Gemma-2b vs. a Python-code finetune; Mistral-7B vs. MetaMath) using reconstruction metrics, residual-activation similarity, and feature-level activation and logit similarity. The transfer result differs sharply between the two pairs, which the author attributes to how much the finetune moved the residual stream.

## Key Findings
- Residual-stream similarity between base and finetune tracks SAE transfer: global mean cosine similarity is 0.67 for Gemma-2b vs. its finetune and 0.96 for Mistral-7B vs. MetaMath (Section 3).
- For Gemma-2b, base SAE reconstruction degrades badly on the finetune (reconstruction CE loss delta 4.21 vs. 0.51 on base; R2 of -10.27), while Mistral-7B stays close to base (loss delta 0.16 vs. 0.15; R2 0.58 vs. 0.68) (Section 4.3).
- Feature density histograms for Mistral are nearly unchanged after finetuning (log-density Pearson correlation 0.94), but Gemma-2b densities diverge (0.47) (Section 4.3).
- On Mistral, regular features have mean activation similarity 0.958 and mean logit similarity 0.9996 with their finetuned counterparts. On Gemma-2b, mean activation similarity is 0.39, with roughly 10% of features above 0.8 (Sections 5.2, 5.3). The author flags these ~10% as possibly noise and untested.
- The author later notes that the logit-similarity numbers mostly reflect that the unembedding barely changed (Frobenius norm ratio 1.117), not that feature directions were preserved. The activation similarity is a statement about encoder directions, not decoder directions (Section 5.2, "Edit").
- Conclusion: transferability is model-dependent and sensitive to the finetuning process; it can be estimated in advance with weight-deviation or activation-similarity heuristics, but this is only anecdotally supported (Section 6.1).

## Methodology
Take the base-model SAE (SAELens releases: gemma-2b-res-jb layer 6, 16k features; mistral-7b-res-wg layer 8, 65k features) and apply it, unchanged, to the finetuned model. Metrics: L0, CE loss with substitution, R2, dead-feature fraction, and density histograms, computed on 256K tokens from the SAE's training distribution. Outlier activations (L2 norm above 290 for Gemma, 200 for Mistral) are filtered for the loss and R2 metrics. Feature-level analysis follows Anthropic's "universality" test (activation and logit correlations per feature) on a sample of up to 1000 regular features (log10 density in [-5, -1]) and 100 exclusively-dead features. Each finetune is a full finetune. Only one residual layer per model is studied, and no SAE is retrained.

## Relevance to Proposal
**Score: 4**

The post is a small, informal study of whether a fixed SAE survives a fine-tune. This touches strand 2 (feature transfer base→fine-tuned, model diffing) and the core question of whether a decomposition built on the base model stays usable after post-training. However, it studies SAEs, not parameter decompositions (VPD/SPD/APD), and it does not attempt any update-aware map Φ(D, δW). Its useful contribution is the explicit distinction between reconstruction quality and feature-level transfer, and the caution that encoder-direction similarity does not imply decoder-direction preservation.

## Limitations
- Only two model pairs, one residual layer each, and no replication across finetunes. The author says generalisation is unclear.
- Each finetune is a full finetune of unknown size and recipe. The author cannot control or vary the update magnitude, so the mechanism behind the Gemma/Mistral difference is not isolated.
- Evaluation uses the SAE's pretraining distribution, not the finetune's target distribution.
- Feature analysis is partly unverified: the author did not manually check the claimed transferring features for interpretability, and the logit-similarity metric is acknowledged to be misleading.
- The author notes that "dead" and "regular" feature counts are sensitive to the outlier-filtering threshold, which was chosen by trial.
- Relies on pretrained SAEs from SAELens, so the SAEs themselves were not designed for the transfer question.
- The referenced code repository was private when a commenter tried to access it; the author later made it public (per comments).

## Key Quotes
> "Now, this is what *bad* SAE transferability looks like!" (Section 4.3)

> "The key point is that the latter is a statement about the feature's encoder direction ... not the decoder one - which is what we mean when we talk about feature directions." (Section 5.2, Edit note)

> "the most obvious answer that comes to mind now is - it depends!" (Section 6.1)

## Comments Summary
Three top-level comments and several replies.

- **faul_sname** (score 9) proposes checking the reverse direction: SAE features on the instruction-tuned model that are dead in the base model, which might show what instruction tuning adds. Suggests Gemma Scope 9B base/instruct residual SAEs as a route. Also proposes learning a single-layer map from residual activations to feature activations as a way to test whether a model "has" a feature.
- **Taras Kutsyk** (replies) agrees the single-SAE setup cannot separate "feature direction changed" from "feature no longer recognised." Proposes training separate SAEs on base and finetune and matching features across them with activation and logit similarity. Doubts that a single-layer probe can be trained meaningfully on mismatched SAE activations.
- **Nathan Helm-Burger** links a related post on better similarity measures for features (no substantive discussion).
- **Roger Dearnaley** (score 4) argues the high cosine similarity for Mistral (0.96) indicates small, targeted changes from standard fine-tuning/DPO, and says the Gemma 2B pair may have been distilled separately. The author expects the Gemma 2B base vs. Instruct gap to be the main driver, consistent with Kissane et al.
