---
slug: moe-dense-crosscoder-2603-05805
title: Sparse Crosscoders for diffing MoEs and Dense models
authors: Chaudhari, Hundia, Gulati
year: 2026
id: 2603.05805
tags: [DIFFING, BASELINE, BACKGROUND]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: arXiv HTML full text (short workshop-style paper)
digest_buckets: []
importance: 2
---

# Sparse Crosscoders for diffing MoEs and Dense models (Chaudhari 2026)

**What it does.** Trains a 5-layer dense model and a 5-layer MoE with equal active parameters from scratch on ~1B
tokens (arXiv, StarCoder code, SimpleStories), then trains a BatchTopK crosscoder with designated shared features on
layer-3 outputs of both and classifies latents by relative decoder norm.

**Key result.** A standard crosscoder labelled many latents "shared" whose two decoders had near-zero cosine. With
designated shared features (and a larger shared-penalty ratio than prior base-vs-fine-tune work used) it reaches 87%
variance explained, and finds 910 MoE-only, 3,226 dense-only and 18,940 shared latents. MoE-only latents are denser than
shared ones and dense-only ones sparser; there is no trimodal norm distribution. No qualitative or causal validation of
the latents.

**Quote.** "This suggests that the standard crosscoder objective overestimates shared structure when the activations
of the compared models differ substantially." (Sec 4)

**Relevance to our paper.** Little direct use: it diffs independently trained models, not a fine-tune. The one
transferable point is that relative-decoder-norm "sharing" is blind to decoder direction (shared by norm, orthogonal by
cosine), a caveat our crosscoder share should state, since a fine-tune that rotates writes could read as shared.
