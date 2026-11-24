---
slug: null-calibrated-sae-reproducibility
title: Null-Calibrated Evaluation of Sparse Autoencoder Decoder Reproducibility
authors: Bright Liu
year: 2026
id: https://icml.cc/virtual/2026/68295
tags: [BASELINE, BACKGROUND]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [baselines, decompositions]
read_from: ICML 2026 workshop page abstract only (no arXiv version found by search; a related ICML 2026 paper by the same author, "Timescale Separation in Sparse Dictionary Learning: Reconstruction Converges Before Reproducibility", icml.cc/virtual/2026/72330, seen only as a search snippet)
digest_buckets: []
importance: 3
---

# Null-Calibrated Evaluation of Sparse Autoencoder Decoder Reproducibility (Liu 2026)

**What it does.** Treats SAE decoder reproducibility across random seeds as a benchmark-design problem: every stability
score is reported against a metric-specific random-dictionary null, pairwise seed statistics are treated as dependent,
and decoder geometry is audited with assignment-based, activation-level, firing-overlap, causal, streaming and
synthetic-ground-truth controls. A workshop poster (Combining Theory and Benchmarks workshop).

**Key result.** In compute-limited cached-activation regimes, reconstruction can look converged while decoder-column
similarity across seeds is within 1.5% of the geometric null; longer training raises decoder agreement, but activation
and functional diagnostics lag.

**Quote.** "reconstruction can appear converged while decoder-column similarity remains within 1.5% of the geometric null" (abstract)

**Relevance to our paper.** It bears on our BatchTopK SAE featurization and on the crosscoder baseline: a reuse reading
at SAE latents assumes the dictionary is a stable object, and this says seed agreement should be checked against a
random-dictionary null before latents are treated as variables. It is also the kind of null our own cross-seed control
for a decomposition would need. Only the abstract was read, so this is a pointer, not a source of numbers.

**For the digest.**
