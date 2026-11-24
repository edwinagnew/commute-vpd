---
slug: kissane2024-saes-transfer
title: "SAEs (usually) Transfer Between Base and Chat Models"
authors: Kissane et al. (Krzyzanowski, Conmy, Nanda)
year: 2024
id: https://www.lesswrong.com/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models
tags: [DIFFING, SUPPORTS, BASELINE]
primary_tag: DIFFING
in_our_bib: kissane2024transfer
cited_in_draft: yes
themes: [model-diffing, baselines]
read_from: "LessWrong post (2024-07-18) read via WebFetch: models, metrics, results tables, failure case, conclusions"
digest_buckets: [4-supports]
importance: 4
---

# SAEs (usually) Transfer Between Base and Chat Models (Kissane 2024)

**What it does.** An interim report that splices SAEs trained on a base model's residual stream into its chat model, and
the reverse, and measures CE loss recovered, MSE and L0. Pairs: Mistral-7B and Mistral-7B-Instruct (layer 16), Qwen1.5
0.5B and its chat model, Gemma v1 2B and its instruct model. It also fine-tunes a base SAE on a few million chat
activations.

**Key result.** On Mistral, the base SAE in the chat model recovers 98.85% of CE loss against 99.01% for a chat SAE, at
a higher L0 (126 vs 101). The chat SAE in the base model recovers 96.82%. Qwen behaves the same. Gemma v1 2B does not
transfer, which the authors put down to its base and chat weights differing unusually much. Fine-tuning the base SAE on
5M chat tokens matches a chat SAE trained on 800M. There is no feature-level check of whether latents keep their meaning.

**Quote.** "SAEs trained on the base model transfer surprisingly well to reconstructing chat activations" (post)

**Relevance to our paper.** It supports fixing the featurization from the base model (our τ fixed from M). The cited
use is fair. It is reconstruction, not mechanism: an SAE that reconstructs chat activations says nothing about whether
the chat model computes those latents the same way, which is our point. The Gemma v1 failure is worth a clause wherever
the draft leans on base featurizations transferring. A base featurization can fail on a fine-tune with a large weight
change, and our definition then has no variables to read.

**For the digest.**
- 4-supports: base SAEs reconstruct chat activations (Mistral 98.85% CE recovered), so a base-fixed featurization is usable; Gemma v1 2B is the stated exception.
