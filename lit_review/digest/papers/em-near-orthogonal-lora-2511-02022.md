---
slug: em-near-orthogonal-lora-2511-02022
title: Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior
authors: Daniel Aarao Reis Arturi et al.
year: 2025
id: 2511.02022
tags: [DIFFING, ALIGNMENT, FT-INTERP, BACKGROUND]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes]
read_from: arXiv HTML full (short workshop paper: abstract, sections 1-5, appendix A)
digest_buckets: []
importance: 2
---

# Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior (Arturi 2025)

**What it does.** Weight-space diffing of emergent-misalignment (EM) LoRAs. It takes the public rank-32, all-adapter
LoRA model organisms of Turner et al. (Qwen2.5-7B and Llama3.1-8B, fine-tuned on extreme sports, risky financial
advice and bad medical advice) and compares the LoRA updates ΔW with each other and with the base weights by cosine
similarity, principal angles and projection overlap. It then interpolates linearly between base and EM weights, and
between EM models, and scores misalignment with a GPT-4o judge.

**Key result.** Cosine similarity between a LoRA update and the base weight is about 1e-3, which the paper reads as the
misalignment occupying "a near-orthogonal subspace"; EM-EM update pairs have mean principal angles ~20° (random ~85°)
and projection overlaps ~0.8, a pattern the appendix says non-EM adapters do not show. Base-to-EM interpolation lowers
alignment near-monotonically (~90% to 55-75%), and cross-task interpolations stay misaligned (~32% / ~16% EM
responses) with >90% coherence.

**Quote.** "EM-base weight pairs are mostly orthogonal while EM-EM pairs exhibit low principal angles and high subspace
overlaps" (Section 1, contributions)

**Relevance to our paper.** Little. Its "near-orthogonal to the base" reading compares an update with the base weight
matrix itself, and a cosine of ~1e-3 is what any low-rank update has against a full-rank matrix in this dimension (its
own random baseline is the same), so it says nothing about whether the update reuses base mechanisms; it should not be
cited as evidence that EM builds something new. The between-fine-tune similarity is a cheap weight-diff reading
(subspace overlap of two updates), not a reuse reading: it says two inducers find the same direction, which fits the
persona-recruitment story of Wang et al. (in our bib) but is not tested against a known answer. No causal test inside
the model.

**For the digest.** (none)
