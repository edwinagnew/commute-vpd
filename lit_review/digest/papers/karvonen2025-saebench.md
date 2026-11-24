---
slug: karvonen2025-saebench
title: "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability"
authors: Karvonen et al.
year: 2025
id: 2503.09532
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: karvonen2025saebench
cited_in_draft: yes
themes: [decompositions, other]
read_from: "arXiv HTML v4: abstract, sec. 3-5 (metrics, results), appendix B (training details), appendix E (random-init baseline)"
digest_buckets: []
importance: 2
---

# SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability (Karvonen 2025)

**What it does.** A benchmark of eight SAE metrics: core (L0, loss recovered), sparse probing, feature absorption,
automated interpretability, RAVEL disentanglement, spurious-correlation removal (SCR), targeted probe perturbation
(TPP) and unlearning. Releases over 200 SAEs of seven architectures (ReLU, TopK, BatchTopK, JumpReLU, Gated, P-anneal,
Matryoshka BatchTopK) trained on the residual stream of Gemma-2-2B (layer 12) and Pythia-160M (layer 8), widths 4k,
16k and 65k, target L0 in {20, 40, 80, 160, 320, 640}, 500M Pile tokens, lr 3e-4, batch 2,048, context 1,024.

**Key result.** Gains on proxy metrics do not reliably translate to practical metrics. Matryoshka BatchTopK slightly
underperforms on the sparsity-fidelity frontier but is best on 5 of 8 metrics in the L0 range 40-200, and its lead grows
with width. Many metrics reach most of their value by 50M tokens. Disentanglement metrics prefer L0 above 400.

**Quote.** "gains on proxy metrics do not reliably translate to better practical performance." (abstract)

**Relevance to our paper.** Cited as the source of our SAE training recipe; the numbers we quote (16,384 latents, k =
80, 500M Pile tokens) are in its grid and Table 3. A departure to list: SAEBench trains on the residual stream at one
middle layer, while ours sit at each block's attention and MLP outputs. Its result that Matryoshka SAEs disentangle
better is a possible alternative SAE featurization if reviewers ask whether the SAE reading depends on architecture.
No bearing on reuse.

**For the digest.**
