---
slug: li2020-greedy-low-rank
title: "Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning"
authors: Li et al.
year: 2020
id: 2012.09839
tags: [DYNAMICS, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics]
read_from: "arXiv HTML: abstract, introduction, related work, Section 3 background, section structure of 4-7"
digest_buckets: []
importance: 2
---

# Towards Resolving the Implicit Bias of Gradient Descent for Matrix Factorization: Greedy Low-Rank Learning (Li 2020)

**What it does.** Theory of the implicit bias of gradient flow on symmetric matrix factorisation W = UU^T with a convex
loss. Shows that gradient flow from infinitesimal initialisation is equivalent to Greedy Low-Rank Learning (GLRL):
optimise under a rank constraint and raise the rank by one, along the top eigenvector of the negative gradient,
whenever the current rank cannot reach a global minimum. Extends to depth ≥ 3.

**Key result.** The equivalence holds under stated assumptions; it yields a counter-example to Gunasekar et al.'s
conjecture that gradient flow finds the minimum-nuclear-norm solution. Depth makes the greedy rank-by-rank behaviour
depend much less on how small the initialisation is, so it is more likely at practical scales.

**Quote.** "gradient flow with infinitesimal initialization is mathematically equivalent to a simple heuristic rank minimization algorithm, Greedy Low-Rank Learning, under some reasonable assumptions." (Abstract)

**Relevance to our paper.** Indirect at most: it describes training from near-zero weights, whereas a fine-tune starts
from a large pretrained initialisation, so it does not predict whether an update rescales or rotates base components.
It could back a sentence on why small updates tend to be low effective rank, if we discuss the effective-rank reading,
but the paper itself does not make that claim about fine-tuning.

**For the digest.**
- (no bucket)
