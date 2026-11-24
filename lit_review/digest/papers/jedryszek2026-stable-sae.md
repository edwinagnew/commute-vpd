---
slug: jedryszek2026-stable-sae
title: "Stable and Steerable Sparse Autoencoders with Weight Regularization"
authors: Jedryszek & Crook
year: 2026
id: 2603.04198
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: abstract, intro, methods (sec. 2), MNIST (sec. 3), Pythia-70M results (sec. 4), discussion"
digest_buckets: []
importance: 1
---

# Stable and Steerable Sparse Autoencoders with Weight Regularization (Jedryszek 2026)

**What it does.** It adds L1 or L2 penalties on SAE encoder and decoder weights and measures cross-seed feature
consistency (Hungarian-matched cosine), auto-interpretability and steering success. The tests are on MNIST and on
layer-3 residual activations of Pythia-70M-deduped (TopK, BatchTopK, Matryoshka, from SAEBench).

**Key result.** On Pythia TopK SAEs, L2 raises the share of strictly shared alive features across three seeds from
under 2% to about 35%. Steering success rises from 6.3% to 13.0%. But L2 kills most latents (1,699 alive of 16,384 at
k=40), and the authors suspect much of the gain is implicit dictionary-size selection.

**Quote.** "adding a small L2 weight penalty increased the fraction of features shared across three random seeds and roughly doubles steering success rates" (Abstract)

**Relevance to our paper.** Little. Seed instability of SAEs is a known caveat for our SAE featurization (BatchTopK
latents), but this paper adds nothing about fine-tuning or diffing.

**For the digest.** (no bucket)
