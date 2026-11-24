---
slug: minder2025-crosscoder-sparsity
title: "Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning"
authors: Minder et al. (Dumas, Juang, Chughtai, Nanda)
year: 2025
id: 2504.02922
tags: [BASELINE, DIFFING, FT-INTERP, DEFINITION]
primary_tag: BASELINE
in_our_bib: minder2025overcoming
cited_in_draft: yes
themes: [model-diffing, baselines, interpreting-finetunes]
read_from: "arXiv HTML (v4): intro, Sec. 2-3 (crosscoder losses, Complete Shrinkage and Latent Decoupling, Latent Scaling, causal KL experiments, template tokens), limitations"
digest_buckets: [1-missing-method, 3-definition]
importance: 7
---

# Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning (Minder 2025)

**What it does.** It shows that L1 crosscoders between Gemma 2 2B base and chat mislabel shared latents as "chat-only",
for two reasons: Complete Shrinkage (L1 drives a small base decoder norm to zero) and Latent Decoupling (a chat-only
latent is rebuilt from other base latents). It introduces Latent Scaling, which fits how well each latent explains the
base model's reconstruction and error, and shows that BatchTopK crosscoders largely avoid both artifacts. Causal tests
patch chat-specific latents into base activations and measure the KL to the chat model's outputs.

**Key result.** Most L1 chat-only latents are not chat-specific, while most BatchTopK ones are. Patching the top 50% of
BatchTopK latents by norm difference cuts the KL on the first 9 tokens by 50%, against 6% for the bottom 50%. The
chat-specific latents include refusal, false-information and personal-question concepts. Much of the behaviour sits at
chat-template tokens, and the crosscoder's error term carries at least as much behaviour as its dictionary.

**Quote.** "a significant limitation is our inability to distinguish between truly novel latents learned during chat-tuning and existing latents that have merely shifted their activation patterns" (Limitations)

**Relevance to our paper.** This is the crosscoder baseline the draft already uses. The limitation quoted above is our
gap stated by the baseline's own authors: cite it in the gap paragraph, since it supports "no test looks inside the
model". Latent Scaling is a reuse-like reading the draft may not be scoring: a "new" latent that a scale of base
latents explains is relabelled as shared, which is close to our kept-up-to-a-gain clause applied to activations. If our
crosscoder share is read without Latent Scaling, the draft should say so, or add it as the authors' own recipe (memory:
baselines use their strongest recipe). Its observation that chat amplifies the norm of shared representations is a
rescaling finding that contrasts with our "turn rather than scale" result on weights.

**For the digest.**
- 1-missing-method: Latent Scaling is the authors' own fix; check our crosscoder baseline uses it, or run it.
- 3-definition: quote the "novel vs merely shifted" limitation as the stated gap our definition fills.
