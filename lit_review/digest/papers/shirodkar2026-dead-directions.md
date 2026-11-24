---
slug: shirodkar2026-dead-directions
title: "Measuring Dead Directions: Decomposing and Classifying Singular Structure off Canonical Alignment"
authors: Shirodkar
year: 2026
id: 2607.00603
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics, other]
read_from: "arXiv HTML: abstract, introduction, Section 4 (taxonomy incl. 4.2 fine-tuned ViT), section headings of 5-9"
digest_buckets: []
importance: 1
---

# Measuring Dead Directions: Decomposing and Classifying Singular Structure off Canonical Alignment (Shirodkar 2026)

**What it does.** Gives a way to read, at one frozen checkpoint, the order k of each "dead" direction in parameter
space (where the Fisher metric degenerates) from how the directional Fisher grows along that direction, giving the
per-direction learning coefficient 1/(2k) of singular learning theory. Directions come from K-FAC joint modes, so no
alignment to coordinate axes is needed; the read separates true singularities from flat gauge symmetries.

**Key result.** On a gelu transformer block with a rotated dead direction the read returns k̂ = 1.95 (true 2) where an
axis-aligned scan reads 1.26. A DINOv2 ViT-S fine-tuned on CIFAR-100 has no dead units; its dead structure is the
LayerNorm-kernel gauge, whose directional Fisher sits 10^4 to 5x10^4 below a live direction.

**Quote.** "We give a descent-free, alignment-free measurement of singular structure on trained networks." (Abstract)

**Relevance to our paper.** No direct bearing: it characterises degenerate directions of a single trained network, not
what a fine-tune changed or reused.

**For the digest.**
- (no bucket)
