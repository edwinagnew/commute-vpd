---
slug: geiger2025-causal-abstraction-foundation
title: "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability"
authors: Atticus Geiger et al.
year: 2025
id: 2301.04709
tags: [DEFINITION, CAUSAL]
primary_tag: DEFINITION
in_our_bib: geiger2025
cited_in_draft: yes
themes: [causal-abstraction, reuse-definitions]
read_from: "arXiv HTML, both halves via WebFetch offsets, queried for the formal definitions (causal model, mechanism, interventions, constructive abstraction, approximate abstraction, distributed interchange interventions); not read line by line"
digest_buckets: [3-definition]
importance: 7
---

# Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability (Geiger 2025)

**What it does.** Gives a general theory of causal abstraction for mechanistic interpretability. It defines:
- causal models, in which each variable's value is set by a mechanism;
- hard interventions, soft interventions and the more general "interventionals";
- exact transformation and constructive abstraction under an alignment between low-level and high-level variables;
- bijective translations, which change the basis, and the distributed interchange interventions built on them;
- approximate abstraction, scored by interchange intervention accuracy.

It argues that the methods of mechanistic interpretability are special cases of this framework.

**Key result.** A theory paper. A high-level model is a constructive abstraction of a low-level model under an alignment if
and only if it is an exact transformation of it. Abstraction is made approximate by the share of interchange interventions
on which the two models agree, with a cutoff η.

**Quote.** "ℋ is a constructive abstraction of ℒ under an alignment ⟨Π,π⟩ iff ℋ is an exact transformation of ℒ." (Def. 36)

**Relevance to our paper.** Our definition is built on this framework: the featurized model, variables, mechanisms F_X,
soft interventions and the interchange test for "needed". It is already cited. The framework relates a low-level model to a
high-level model, and does not define when two low-level networks (M and F) under one featurization "have the same
mechanism". Our kept clause (same mechanism up to a positive scalar on the parent settings that occur) is therefore a new
relation, not one inherited from Geiger. The draft should say this plainly in its list of departures. The same holds for ρ:
it is a write share, not the paper's interchange intervention accuracy. The paper does not discuss fine-tuning.

**For the digest.**
- `3-definition`: the vocabulary and the interchange test we use. Our kept relation between two models' mechanisms and our ρ are departures, and the draft should state both.
