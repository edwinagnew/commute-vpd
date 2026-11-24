---
slug: wang2026-lowrank-implicit-reg
title: "Stability of Low-Rank Implicit Regularization in Perturbed Deep Matrix Factorization"
authors: Wang & Chou
year: 2026
id: 2605.28613
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: arXiv HTML abstract, introduction and contributions; theorem sections skimmed by heading
digest_buckets: []
importance: 0
---

# Stability of Low-Rank Implicit Regularization in Perturbed Deep Matrix Factorization (Wang 2026)

**What it does.** Theory for gradient descent on deep matrix factorisation of a symmetric target. It gives sufficient
spectral conditions (target spectrum, initialisation, step size) for a non-empty "low-rank interval" along the
trajectory, then extends convergence and low-rank-interval results to an additively perturbed target.

**Key result.** Convergence guarantees and eigenvalue recovery bounds under perturbation, and a proof that the
iterates' effective rank stays near that of the noiseless target's rank-L approximation over a perturbed interval,
with explicit dependence on the perturbation size; small numerical illustrations.

**Quote.** "the effective rank of the iterates remains close to that of the rank-L approximation of the noiseless target over a perturbed low-rank interval" (abstract)

**Relevance to our paper.** None in practice. It is matrix-factorisation optimisation theory with no language model,
fine-tune or mechanism; it could only be a distant theory pointer for why updates come out low-rank.

**For the digest.** (none)
