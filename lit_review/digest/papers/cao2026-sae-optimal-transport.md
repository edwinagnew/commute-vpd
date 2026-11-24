---
slug: cao2026-sae-optimal-transport
title: "Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression"
authors: Cao et al.
year: 2026
id: 2605.28567
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: arXiv HTML abstract, sections 1-3, 5 and 6 (experiments, case studies); theory section 4 skimmed by heading
digest_buckets: []
importance: 1
---

# Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression (Cao 2026)

**What it does.** Represents each SAE feature by the activation-weighted distribution of the hidden states (or their
images in a shared reference space) on its top-K activating tokens, and compares features by Wasserstein distance.
Used to match features across layers (GPT-2 TopK SAEs, Gemma-2-2B Gemma Scope MLP SAEs) and to cluster circuit nodes
into supernodes.

**Key result.** Beats decoder-vector matching (SAE Match, Feature Flow) on LLM-judged match accuracy across distant
layers while staying competitive on CE and variance explained under feature swap; compresses Marks-style feature
circuits into supernodes that match hand-made ones (the "Dallas" circuit) and separates digit-addition features
by their true function in a blind human ranking.

**Quote.** "each feature is represented not by a single decoder vector like in the literature, but by an activation-weighted distribution over the hidden states that express it." (abstract)

**Relevance to our paper.** Little direct bearing. It works within one model; the idea of matching features by where
they fire, through token-aligned states, could in principle match base and fine-tuned SAE features, but it compares
by value, and our definition compares mechanisms, not values. Not needed in the draft.

**For the digest.** (none)
