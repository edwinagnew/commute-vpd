---
slug: franco2026-attn-singular-vectors
title: "Singular Vectors of Attention Heads Align with Features"
authors: Franco et al.
year: 2026
id: 2602.13524
tags: [DECOMP, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [decompositions]
read_from: "arXiv HTML: abstract, sec. 1-4 (toy model, theorems), sec. 5 (sparse attention decomposition, Pythia-160M checkpoints)"
digest_buckets: []
importance: 2
---

# Singular Vectors of Attention Heads Align with Features (Franco 2026)

**What it does.** Asks why the singular vectors of a head's QK matrix W_Q^T W_K often line up with features. It uses a
toy superposition autoencoder with an added attention head trained to attend when given feature pairs are present.
It proves that a head trained for one feature pair becomes rank one, with singular vectors equal to the
covariance-whitened features, and that features of interest orthogonalise against the rest. It then tests one
prediction, sparse attention decomposition, in GPT-2 and over Pythia-160M training checkpoints on IOI heads.

**Key result.** In the toy model singular vectors align with features, one singular value per attended pair, and the
singular values track the target logits. In Pythia-160M, attention on IOI token pairs becomes sparse in the SVD basis
during training, and not under randomly rotated singular vectors.

**Quote.** "singular vectors robustly align with features in a model where features can be directly observed" (abstract)

**Relevance to our paper.** Little. It supports a cheap weight-derived featurization of heads (QK singular vectors),
which could serve as a featurization fixed from M if we test whether a copy fine-tune keeps or rotates a head's
directions, the open "rotation-tolerant kept clause". It says nothing about fine-tuning, diffing or reuse.

**For the digest.** (none)
