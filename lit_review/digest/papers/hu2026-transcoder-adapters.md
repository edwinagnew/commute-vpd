---
slug: hu2026-transcoder-adapters
title: Transcoder Adapters for Reasoning-Model Diffing
authors: Hu, Ward, Icard, Potts
year: 2026
id: 2602.20904
tags: [BASELINE, FT-INTERP, DIFFING, KNOWN-ANSWER, SUPPORTS, CITE-FIX]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes, baselines, elicitation]
read_from: arXiv PDF pages 1-9 (full main text); appendices not read
digest_buckets: [1-missing-method, 2-known-answer-case, 4-supports, 5-cite-fix]
importance: 6
---

# Transcoder Adapters for Reasoning-Model Diffing (Hu 2026)

**What it does.** Learns, at each layer, a sparse transcoder that runs beside the frozen base MLP so that base MLP +
adapter reproduces the fine-tuned MLP, inside a replacement model that keeps the fine-tune's attention. Trained with
output KL, normalized MSE and "bridging" KL losses on 50k OpenThoughts3 samples, for Qwen2.5-Math-7B (base) vs
DeepSeek-R1-Distill-Qwen-7B (target); 8192 latents per layer, L0 0.1-10. A "hybrid" control (target model with base
MLPs) checks that attention changes alone do not explain the behaviour.

**Key result.** Even the sparsest adapter cuts top-1 disagreement with the target from 28.5% (hybrid) to 13.6% (9.2%
at L0 10; MLP-fine-tune skyline 8.5%), matches the target's response lengths and recovers typically 50-90% of the
accuracy gain; the hybrid shows little reasoning. An LLM judge calls only 8.6% of adapter features reasoning-specific,
48% general language and 37% domain (maths, code, science). Hesitation ("Wait") traces to 5.6k features of 229k (4.8k
output features plus 811 template features): removing them cuts "wait" per 1k tokens from 5.5 to 0.5 and response
length from 7.8k to 3.5k, mostly without accuracy loss; adding both groups to the hybrid raises it to 8.2, and 50,000
random features do less. Hesitation features get their input mostly straight from token embeddings, which the authors
read as reliance on base representations (citing Ward et al.).

**Quote.** "a feature's activating examples suggest what inputs trigger new computation—not whether the underlying
representations are unique to the target model or already present in the base model." (Sec 5)

**Relevance to our paper.** This is a model-diffing method we neither run nor cite, and it is the nearest published
tool to ours: it decomposes the update's computation (not activations) into sparse units and validates them by
ablation and addition. Unlike our definition it makes no reuse call, since every adapter feature is a change by
construction, and the quote says it cannot tell new from already present. It could run on the 67M (four MLPs, cheap)
as a baseline: score reuse as the share of the needed change carried by adapter features whose encoder reads base
latents, then test it on the nine built pairs. As a realistic case it is only partly known-answer: the hybrid control
shows the MLP changes are needed, and the attribution graphs suggest the new hesitation computation reads base
representations; that is "new mechanism on reused variables", a mixed case our definition should handle. The 8.6%
figure and the "domain knowledge" reading weakly support re-weighting over new reasoning machinery; the draft should
cite it under model diffing and next to Ward et al.

**For the digest.**
- 1-missing-method: sparse adapter on the MLP difference, validated by ablate/add; runnable on the 67M; no reuse score as published.
- 2-known-answer-case: R1-Distill-Qwen-7B vs Qwen2.5-Math-7B, hesitation; MLP change needed (hybrid control), new features read base embeddings; partial evidence only.
- 4-supports: only ~8% of change features are reasoning-specific, most look like domain or general language features.
