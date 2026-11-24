---
slug: lundstrom2026-catastrophic-forgetting
title: "Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning"
authors: Lundstrom-Imanov
year: 2026
id: 2601.18699
tags: [FT-INTERP, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, other]
read_from: arXiv HTML (current version) abstract, sections 1-9: methods, cohort, results, LRCP, limitations
digest_buckets: []
importance: 0
---

# Mechanistic Analysis of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning (Lundstrom-Imanov 2026)

**What it does.** A single-author study of forgetting under sequential fine-tuning (code, then maths, then medical QA)
on 20 models: 10 closed API models (Claude Fable 5, GPT-5.5 High, Gemini 3.5 Flash and others), scored by an output
"semantic drift", and 10 open-weight models (DeepSeek-V4-Pro, Kimi K2.6, Llama 4, Qwen 3.6 and others), scored by
CKA between base and adapted activations, Frobenius weight drift, MoE routing drift and activation patching. It
proposes Low-Rank Circuit Projection (LRCP): gradients projected off the top SVD subspace of old-task activations.

**Key result.** Mid-layer CKA falls to 0.41-0.61; MLP down-projections drift most; early attention heads grow more
diffuse. On Llama 4 Scout, LRCP is reported to keep code accuracy at 74.8% against 35.1% for plain fine-tuning, with
maths nearly unchanged.

**Quote.** "LRCP selectively restricts gradient updates to the orthogonal complement of the subspace spanned by historical task activation states." (§1)

**Relevance to our paper.** None worth citing. The setting is forgetting, not reuse, and the measures are global
drift statistics. The paper also has credibility problems: it says all 20 models, closed API models included, go
through the same fine-tuning pipeline, which is not possible for most of them; it fine-tunes 1T-1.6T-parameter MoEs
with no compute stated; its schedule is "a constant learning rate ... with a cosine decay"; and no seeds or variances
are given. Its numbers should not be relied on.

**For the digest.** (none)
