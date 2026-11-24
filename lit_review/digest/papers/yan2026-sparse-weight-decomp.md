---
slug: yan2026-sparse-weight-decomp
title: "Sparse Weight Decomposition for Efficient Circuit Extraction"
authors: Yan et al.
year: 2026
id: 2608.03913
tags: [DECOMP, CAUSAL, BASELINE]
primary_tag: DECOMP
in_our_bib: no
cited_in_draft: no
themes: [decompositions, causal-abstraction, baselines]
read_from: "arXiv HTML: abstract, sections 1-3.4 (method, baselines, evaluation protocol, single-matrix and full-model replacement); 3.5-3.7 and appendices skimmed by grep only"
digest_buckets: [1-missing-method]
importance: 3
---

# Sparse Weight Decomposition for Efficient Circuit Extraction (Yan 2026)

**What it does.** Factorises each pretrained weight matrix as W ≈ AB with A and B sparse (adapted from Double Sparse
Factorization), so each intermediate coordinate is a fixed rank-one "bottleneck unit" with few read and write edges.
Units are ranked by attribution patching and kept or mean-ablated to build task circuits (greater-than, IOI,
docstring, gendered pronoun). It is compared at matched replacement loss with Transcoders, VPD (original KL objective,
and a variant with local reconstruction) and weight-sparse pretraining, on GPT-2 small, Qwen2.5-0.5B to 3B and
Qwen3.5-27B. There is also a zero-data variant fitted from the weights alone.

**Key result.** SWD reaches low CE delta after a few thousand calibration tokens, against about 10^6 for the trained
baselines. At matched CE it reaches the same circuit sufficiency and necessity with fewer active edges than
Transcoders and VPD. Original VPD (KL objective) "does not reach the fidelity required" for CE-matched comparison on a
single matrix. Full-model GPT-2 replacement reaches CE 3.44 after fixed-support fine-tuning (dense 3.45), on about 1%
of the sparse-pretraining token budget.

**Quote.** "Without training a separate replacement network, this parametric representation supports the same
scoring, selection, and ablation circuit extraction workflow used for methods that learn sparse features." (Abstract)

**Relevance to our paper.** It is a cheap parameter-space featurization with input-independent units, unlike VPD's
gated subcomponents. That matters for diffing: a factorisation fitted to the base model's W could carry F's update
onto fixed sparse units (by refitting the factor values on a fixed support), which is close to our least-squares carry
onto VPD. Being nearly free (zero-data in the limit), it could serve as a second parameter featurization at Qwen scale,
where VPD's re-gating saturates. Its finding that VPD's KL objective fails single-matrix fidelity is another
independent report of VPD's scaling trouble (compare ASPD). It does not touch fine-tuning, so at most a
related-work line on decompositions.

**For the digest.**
- 1-missing-method: a near-free sparse weight factorisation that could be a second parameter featurization for the reuse definition, especially on Qwen; not a diffing method as published.
