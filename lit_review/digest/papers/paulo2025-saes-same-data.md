---
slug: paulo2025-saes-same-data
title: "Sparse Autoencoders Trained on the Same Data Learn Different Features"
authors: Paulo & Belrose
year: 2025
id: 2501.16615
tags: [BACKGROUND, CITE-FIX]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [decompositions, other]
read_from: "arXiv HTML (v2): abstract, intro, Hungarian-matching method, main results, related work"
digest_buckets: [5-cite-fix]
importance: 3
---

# Sparse Autoencoders Trained on the Same Data Learn Different Features (Paulo 2025)

**What it does.** It trains SAEs on the same model and data that differ only in the random seed, matches latents across
seeds with the Hungarian algorithm on encoder and decoder cosine, and counts a latent as shared above 0.7 on both. It
covers several layers of three LLMs (Pythia 160M, Llama 3 8B and another), two datasets and several SAE architectures.

**Key result.** For a 131K-latent SAE on a Llama 3 8B MLP, only 30% of latents are shared across seeds. Smaller models
and smaller SAEs share more. ReLU SAEs with L1 are more seed-stable than TopK SAEs at matched sparsity. "Orphan" latents
that appear in one seed only can still have good explanations.

**Quote.** "the set of features uncovered by an SAE should be viewed as a pragmatically useful decomposition of activation space, rather than an exhaustive and universal list of features “truly used” by the model." (Abstract)

**Relevance to our paper.** Our definition is relative to the featurization, and one featurization is a BatchTopK SAE,
which is in the TopK family the paper finds most seed-dependent. So an SAE-featurization reading of ρ may move with the
SAE seed. A one-line citation where the draft says reuse is relative to the featurization would help, and a second-seed
check would show whether the SAE row is stable. It does not bear on VPD directly.

**For the digest.**
