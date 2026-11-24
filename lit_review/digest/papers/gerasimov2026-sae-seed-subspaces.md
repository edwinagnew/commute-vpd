---
slug: gerasimov2026-sae-seed-subspaces
title: "Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders"
authors: Gerasimov et al.
year: 2026
id: 2606.12138
tags: [CITE-FIX, BACKGROUND]
primary_tag: CITE-FIX
in_our_bib: no
cited_in_draft: no
themes: [decompositions, reuse-definitions]
read_from: arXiv HTML abstract+intro+preliminaries+methodology+sections 5-7 (main text); appendices only skimmed by grep
digest_buckets: []
importance: 4
---

# Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders (Gerasimov 2026)

**What it does.** Trains many SAEs (up to 96 seeds in places) on identical activations and data, varying only the seed,
and gives each feature a "stability": the estimated probability that another seed's SAE has a decoder vector with
cosine ≥ θ (0.7) to it. Features with stability ≤ 0.05 are "unstable", ≥ 0.95 "stable". It compares the two groups by
frequency, token content, auto-interpretation scores and masking impact on explained variance and next-token loss,
then analyses the effective rank and cross-seed overlap of the subspaces each group spans, with a synthetic low-rank
model. Settings: TopK SAEs mainly on GPT-2 layer 7 (F = 2^14), plus Pythia and Gemma-2 layers, other dictionary sizes,
and BatchTopK, JumpReLU, ReLU+L1 and HierarchicalTopK variants.

**Key result.** Stable features carry most of the reconstruction and loss signal; unstable ones fire less often (0.18%
vs 0.44% of tokens), are mostly surface-form triggers, and masking 4N of them hurts less than masking N stable ones.
Unstable features are individually non-reproducible but span low-rank subspaces that reproduce across seeds. TopK and
BatchTopK are about equally unstable at EV ≈ 0.892; ReLU+L1 is near-fully stable at lower EV. More training leaves a
plateau of ~8% unstable features.

**Quote.** "unstable features are individually non-reproducible but concentrate in reproducible lower-rank subspaces,
suggesting that seed dependence often reflects basis ambiguity within a shared region of activation space rather than
pure noise." (Abstract)

**Relevance to our paper.** Our definition is relative to the featurization, and one featurization we use is BatchTopK
SAE latents, the class this paper finds seed-dependent at the latent level. Two consequences. (i) A per-latent kept or
needed reading on an SAE featurization may change with the SAE seed; a reviewer can ask for a seed check, and this
paper is the citation that motivates it (it does not test fine-tunes, so it does not tell us how big the effect is on ρ).
(ii) Its "basis ambiguity within a reproducible subspace" is the same structure a rotation-tolerant kept clause would
forgive: if the needed latents of a fine-tune turn within a base subspace, a subspace-level kept test (as in this
paper's overlap analysis) is a natural candidate for the pending rotation-tolerant clause. No bearing on the
elicitation question itself.

**For the digest.** (none: methodological caveat for the SAE featurization; cite where the draft says reuse is relative to the featurization)
