---
slug: neyshabur2020-what-is-transferred
title: "What is being transferred in transfer learning?"
authors: Neyshabur et al.
year: 2020
id: 2008.11687
tags: [BACKGROUND, DIFFING, DEFINITION]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, model-diffing]
read_from: "local PDF text (~/commute-data/lit_review_pdfs/survey/2008.11687), pp. 1-9: abstract, contributions, Sec. 3.1 (block shuffling), 3.2 (CKA Table 1, l2 Table 2), 3.3 (basins, module criticality); appendix not read"
digest_buckets: []
importance: 3
---

# What is being transferred in transfer learning? (Neyshabur 2020)

**What it does.** Studies ResNet-50 fine-tuned from ImageNet (P-T) vs trained from random init (RI-T) on CheXpert and
DomainNet (real, clipart, quickdraw). It separates "feature reuse" from low-level statistics by block-shuffling target
images, and compares models by mistakes, per-layer CKA, l2 distance in weights, linear-interpolation loss barriers
(a formal "basin" definition) and an extended module criticality.

**Key result.** Block shuffling shrinks the benefit of transfer on real and clipart but not on quickdraw, and the
optimization speed-up of P-T survives shuffling, so part of what transfers is not visual features. Two P-T models
share a loss basin (no barrier), make similar mistakes and are closer in weights than two RI-T models (CheXpert l2
200 vs 255; DomainNet ~180-220 vs ~780-820, Table 2). Per-layer CKA between the fine-tuned and pretrained model falls
with depth, from 0.62 at conv1 to 0.045 at layer 4 (Table 1). Higher layers are more critical to perturbation.

**Quote.** "we separate the effect of feature reuse from learning low-level statistics of data and show that some
benefit of transfer learning comes from the latter." (Abstract)

**Relevance to our paper.** Background for the "reuse" vocabulary in transfer learning; it gives no definition of
reuse, only proxies (CKA, l2, basin). One number is worth knowing: after fine-tuning, the top block's features have
almost no CKA with the pretrained model's (0.045) even though the fine-tunes stay in one basin near it, so "stays
close in weights" and "keeps its features" come apart in the higher layers. That fits, loosely, our finding that
trained fine-tunes change what heads write rather than rescale it, but it is vision, full fine-tuning on a far
domain, and activation similarity, so it cannot be cited as support for our claim. Not a method to compare against.

**For the digest.**
- (no bucket)
