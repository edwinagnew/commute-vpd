---
slug: braun2025-apd-paper
title: "Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition"
authors: Braun et al. (Bushnaq, Heimersheim, Mendel, Sharkey)
year: 2025
id: 2501.14926
tags: [DECOMP, DEFINITION, BACKGROUND]
primary_tag: DECOMP
in_our_bib: braun2025apd
cited_in_draft: yes
themes: [decompositions, reuse-definitions]
read_from: "arXiv abstract; arXiv HTML v4 Sec. 2 (method, 'mechanism space' definition), Sec. 3 toy results (skimmed), Sec. 4.2 limitations, App. A.1 excerpt"
digest_buckets: [3-definition]
importance: 3
---

# Interpretability in Parameter Space: Attribution-based Parameter Decomposition (Braun 2025)

**What it does.** Introduces APD, which decomposes a network's parameter vector into components that sum to it
(faithful), of which few are needed per input (minimal: batch top-k on gradient attributions, then a second forward pass
with only those components), and which are low rank (simple: a Schatten quasi-norm). It is tested on toy models with
known mechanisms: a toy model of superposition, and compressed computation in one and in two layers.

**Key result.** APD recovers the ground-truth mechanisms "up to a small error" in all three toys: one component per
superposed feature, and one per computation even across neurons or layers. The paper says it is "currently only
appropriate for studying toy models because of its computational cost and hyperparameter sensitivity". It describes
some mixing of mechanisms, which it attributes to top-k.

**Quote.** "we define 'mechanisms' as vectors in parameter space" (Sec. 2)

**Relevance to our paper.** This is the ancestor of the VPD featurization we use, already cited. Its only extra use for
us is as a definition to contrast: here a mechanism is a parameter vector, a component summing with the others to
θ, which is the sense in which a fine-tune's update could "reuse" a component by moving along it. Our definition instead
takes mechanism in Geiger's sense, a variable's function of its parents, at a featurization fixed from M. The draft
should state the difference once, since readers from this line will read "kept up to a gain" as "the update is a scalar
multiple of the component". Little else bears on fine-tuning.

**For the digest.**
- 3-definition: "mechanism = vector in parameter space" (APD Sec. 2.1); contrast with our Geiger mechanism f_u over a fixed featurization.
