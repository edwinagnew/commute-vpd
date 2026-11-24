---
slug: olah2020-zoom-in
title: "Zoom In: An Introduction to Circuits"
authors: Chris Olah et al.
year: 2020
id: https://distill.pub/2020/circuits/zoom-in
tags: [BACKGROUND, CAUSAL, DEFINITION]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, other]
read_from: "Distill article HTML (text stripped): the three claims, Claim 1-3 sections, curve-detector circuit, universality discussion; figures not viewed"
digest_buckets: [3-definition]
importance: 3
---

# Zoom In: An Introduction to Circuits (Olah 2020)

**What it does.** This Distill essay opens the Circuits thread. Mostly from InceptionV1, it argues three "deliberately
speculative" claims. Features are the fundamental units, and correspond to directions (linear combinations of neurons).
Circuits are features connected by weights, i.e. computational subgraphs. Universality holds: analogous features and
circuits form across models and tasks. Its examples include curve detectors built from earlier curve and line detectors,
read off the weights.

**Key result.** No quantitative result. The weights between understood features are themselves interpretable ("You can
literally read meaningful algorithms off of the weights"). The essay offers universality as a hypothesis the authors
"suspect" is true, on low-level vision evidence.

**Quote.** "Features are connected by weights, forming circuits." (Claim 2)

**Relevance to our paper.** This is background. It is the origin of the circuits vocabulary and, per Saphra & Wiegreffe,
of the term "mechanistic interpretability". Two points are worth a sentence. First, its circuit is a set of features plus
the *weights* between them, which fits our weight-level featurization (VPD subcomponents). There a variable's mechanism
is set by weights, and "kept" asks whether those weights still compute the same function. Second, its universality claim
(analogous features across models) is the cross-model cousin of our question. It is about independently trained models
converging, not about a fine-tune keeping its parent's mechanisms, and it gives no test for "same mechanism". There is no
fine-tuning content.

**For the digest.**
- `3-definition`: the source definitions of feature and circuit (features joined by weights), and of the universality hypothesis. It can be cited for vocabulary, not for evidence.
