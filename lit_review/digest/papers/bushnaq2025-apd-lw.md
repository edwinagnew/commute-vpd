---
slug: bushnaq2025-apd-lw
title: Attribution-based parameter decomposition (LessWrong post)
authors: Bushnaq et al. (Braun, Heimersheim, Mendel, Sharkey)
year: 2025
id: https://www.lesswrong.com/posts/EPefYWjuHNcNH4C7E/attribution-based-parameter-decomposition
tags: [DECOMP, BACKGROUND]
primary_tag: DECOMP
in_our_bib: no
cited_in_draft: no
themes: [decompositions]
read_from: "LessWrong post full (via LW GraphQL)"
digest_buckets: []
importance: 2
---

# Attribution-based parameter decomposition, LessWrong linkpost (Bushnaq 2025)

**What it does.** This is the Apollo linkpost for the APD paper (Braun et al. 2025, separate entry). It explains why the
authors chose to decompose parameters rather than activations: each "mechanism" is a vector in parameter space, and the
components must sum to the weights (faithful), be few per input (minimal, via a batch top-k over gradient attributions),
and be low rank (simple, via a Schatten quasi-norm). It also gives their next steps.

**Key result.** On three toy models with known mechanisms (toy model of superposition, and compressed computation in
one and in two layers), APD "is largely able to decompose these toy networks", but it needed "a lot of hyperparameter
tuning and still exhibit[s] some amount of mechanism mixing", which they blame on top-k. The future work listed, learned
masks in place of attributions and rank-one components grouped afterwards, is what SPD and VPD later did.

**Quote.** "we think it might be better to directly decompose a network's parameters into components by parametrising
each mechanism individually." (Motivation)

**Relevance to our paper.** Background only: it is the blog version of a paper the draft already cites (braun2025apd),
and it adds no result on fine-tuning. Cite the paper, not the post. One useful phrase is that the authors already call
circuits "mechanisms", which fits the draft's mechanism vocabulary, but the draft's sense of mechanism comes from Geiger
(a variable's function of its parents), not a parameter vector. If both senses appear, the draft should say which it
means.

**For the digest.**
