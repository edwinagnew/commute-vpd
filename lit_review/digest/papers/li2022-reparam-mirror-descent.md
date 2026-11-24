---
slug: li2022-reparam-mirror-descent
title: "Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent"
authors: Li et al.
year: 2022
id: 2207.04036
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: abstract, sec. 1, sec. 4.1 (Def. 4.1, Examples 4.4-4.5, Cor. 4.12), conclusion"
digest_buckets: []
importance: 2
---

# Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent (Li 2022)

**What it does.** Theory. It defines a "commuting parametrization" w = G(x): the Lie brackets of the gradients of G's
coordinates vanish. It proves that gradient flow on x with such a G equals mirror flow on w, for a Legendre function set
by G and the initialisation, and the converse through the Nash embedding theorem. No experiments.

**Key result.** Separable parametrizations such as w = u⊙u - v⊙v (diagonal, gain-like) commute and give entropy-type
implicit biases. Matrix factorizations UU^T and UV^T do not commute, and gradient flow on them cannot in general be
written as mirror flow (Cor. 4.12). It also gives convergence proofs that earlier work assumed.

**Quote.** "any gradient flow (i.e., the continuous analog of gradient descent) with a commuting parametrization is equivalent to a mirror flow" (conclusion)

**Relevance to our paper.** Little. "Commuting" here is a Lie-bracket property, unrelated to the project's original sense
of commuting an update past a decomposition; avoid the word near it. The one point of contact is that a gain-only
(diagonal) reparametrization, like our IA3-style refits, has a clean implicit bias while factorized (LoRA-like or
rank-one subcomponent) updates do not. That is a fact about optimisation, not about whether a mechanism is reused.

**For the digest.** (none)
