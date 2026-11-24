---
slug: lenc2015-equivariance-representations
title: "Understanding image representations by measuring their equivariance and equivalence"
authors: Lenc & Vedaldi
year: 2015
id: 1411.5908
tags: [DEFINITION, BACKGROUND]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions]
read_from: "arXiv PDF (1411.5908v2, CVPR 2015) via fleet/pdf-text, all 9 pages: abstract, Sec. 1-2 (definitions), Sec. 3.3 (equivalence, Table 4), Sec. 4. Identified from Bansal et al. 2021, which credits stitching to it"
digest_buckets: [3-definition]
importance: 3
---

# Understanding image representations by measuring their equivariance and equivalence (Lenc 2015)

**What it does.** Defines three properties of a representation φ: equivariance (a map M_g with φ(gx) ≈ M_g φ(x) for an
input transform g), invariance (M_g = identity), and equivalence (a map E with φ'(x) ≈ E φ(x) between two
representations). The maps are learned from data; for CNNs equivalence is tested with a learned "stitching layer" that
plugs the first layers of one network into the rest of another, giving "Franken-CNNs". Applied to HOG and to AlexNet,
a reimplementation trained on ImageNet, and Places-trained networks.

**Key result.** With the identity in place of a stitching layer, swapped layers give over 99% top-1 error, so
separately trained networks are not directly compatible. With a learned stitch, ImageNet-trained networks are nearly
interchangeable up to Conv4 (top-1 error 0.43-0.46 vs 0.43 for AlexNet itself), Conv5 less so (0.50), and the
Places-trained network much less at depth (0.65 at Conv5, Table 4).

**Quote.** "Equivalence is then obtained if the resulting “Franken-CNNs” perform as well as the original ones"
(Sec. 1)

**Relevance to our paper.** The origin of model stitching and of a formal "same information up to a learned map"
equivalence, two networks at a time. Like Bansal et al., it compares representations by value, up to an arbitrary
linear map, where we compare mechanisms up to a positive scalar; cite it, if at all, together with Bansal et al. as the
source. Vision only, separately trained networks, not a fine-tune: no known-answer use.

**For the digest.**
- 3-definition: equivalence φ' ≈ Eφ via a learned stitch: the origin of the "up to a linear map" reading of shared representations, to contrast with our up-to-scalar kept clause.
