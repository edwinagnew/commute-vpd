---
slug: brzozowski2026-aligned-training-sae
title: "Aligned Training: A Parameter-Free Method to Improve Feature Quality and Stability of Sparse Autoencoders (SAE)"
authors: Brzozowski et al.
year: 2026
id: 2605.18629
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: abstract, introduction, Section 2, Sections 4.1-4.4 (setup and results)"
digest_buckets: []
importance: 1
---

# Aligned Training: A Parameter-Free Method to Improve Feature Quality and Stability of Sparse Autoencoders (Brzozowski 2026)

**What it does.** Observes that the inner product between each SAE latent's encoder row and decoder column (the
"alignment score") is bimodal, with a large group of near-zero, uninterpretable latents. Reparameterises the SAE so
this inner product equals one for every latent. Trains ReLU, TopK and BatchTopK SAEs on Pythia-160M layer 8 and
Gemma-2-2B layer 12 and scores them on SAEBench.

**Key result.** Pareto improvements in recovered cross-entropy and explained variance at matched L0 for all
dictionary sizes tested (4K, 16K, 65K); dead latents fall from about 20% (ReLU, 4K) to near zero, including for
TopK and BatchTopK; cross-seed dictionary agreement (mean max cosine) improves throughout training.

**Quote.** "a substantial fraction are nearly orthogonal artifacts with alignment scores near zero." (Introduction)

**Relevance to our paper.** Little bearing. Our BatchTopK SAE featurization could use it to cut dead latents and
seed dependence, but nothing in our claims turns on SAE training quality.

**For the digest.**
- (no bucket)
