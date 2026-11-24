---
slug: turner2025-model-organisms-em
title: "Model Organisms for Emergent Misalignment"
authors: Turner et al.
year: 2025
id: 2506.11613
tags: [ALIGNMENT, FT-INTERP, DYNAMICS]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, training-dynamics]
read_from: "arXiv HTML: abstract, sections 1-6 (datasets, robustness across models and protocols, rank-1 LoRA, phase transition, discussion); appendices not read"
digest_buckets: []
importance: 4
---

# Model Organisms for Emergent Misalignment (Turner 2025)

**What it does.** Builds cleaner emergent-misalignment (EM) fine-tunes from three GPT-4o-generated narrow datasets
(extreme-sports advice, risky financial advice, bad medical advice). It tests them across Qwen2.5 (0.5B to 32B),
Gemma-3 (4B to 27B) and Llama-3.1/3.2 (1B, 8B) with all-layer LoRA, full SFT, and a single rank-1 LoRA on the layer-24
MLP down-projection of Qwen2.5-14B. It then tracks that adapter's B vector over training.

**Key result.** Close to 40% EM with over 99% coherence on Qwen2.5-32B (against 6% EM and 67% coherence for the
original insecure-code model). EM appears at 0.5B (up to 8%) and under full SFT (9% to 36% on Qwen-14B after one epoch).
A single rank-1 adapter gives 9.5% to 21.5% EM. The B vector's direction rotates sharply around step 180, with a
gradient-norm peak, while its norm grows smoothly. Scaling the adapter shows EM appearing over a narrow window that
coincides with the rotation.

**Quote.** "we find that a single rank-1 LoRA adapter is sufficient to induce EM." (Section 3.5)

**Relevance to our paper.** This paper is not itself a known-answer case: it shows no base-model mechanism behind the
behaviour. But it supplies the realistic, minimal fine-tunes on which follow-ups give that evidence: Soligo et al. find
a convergent misalignment direction, and Wang et al. (in our bib) show base-model persona features control EM. A rank-1
down_proj adapter on Qwen2.5-0.5B or 1.5B is a cheap, realistic update for our Qwen line, with outside evidence that it
recruits pre-existing persona features. That makes it the closest available candidate known answer for "reuse" on a
realistic fine-tune. The B-direction rotation is a training-trajectory result, not a base-versus-fine-tune mechanism
comparison, so it does not bear on our rescale-or-rotate finding. Cite with Wang et al. if EM is used as a test case.

**For the digest.**
- (no bucket by its own evidence; flag as the source of rank-1 EM fine-tunes that, with Wang et al. 2025, could serve as a realistic reuse known-answer on Qwen)
