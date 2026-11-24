---
slug: beckers2019-abstracting-causal-models
title: "Abstracting Causal Models"
authors: Sander Beckers et al.
year: 2019
id: 1812.03789
tags: [DEFINITION, CAUSAL]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, reuse-definitions]
read_from: "PDF text (v4, AAAI 2019), local copy: main text pp1-8 (definitions 3.1-3.19, examples 3.4 and 3.10, conclusion); proofs not read"
digest_buckets: [3-definition]
importance: 3
---

# Abstracting Causal Models (Beckers 2019)

**What it does.** It gives a chain of ever stricter definitions of when one causal model abstracts another. It starts
from Rubenstein et al.'s exact transformation, then defines uniform transformation (no hiding differences behind a chosen
distribution), τ-abstraction (the allowed interventions are induced by the state map τ), strong τ-abstraction (all
high-level interventions allowed) and constructive τ-abstraction (τ acts blockwise on a partition of the low-level
variables). It is general causal modelling: there are no neural networks in it.

**Key result.** A theory paper. Exact transformation is too weak: two unrelated models can be exact transformations of
each other (Example 3.4). Constructive abstraction is "the cleanest" notion, and the micro-to-macro aggregation examples of
Rubenstein et al. are all strong abstractions. An approximate notion is left to future work.

**Quote.** "So it seems that RW+ would want to claim that if there exists an exact transformation from M1 and M2, then
M2 and M1 are the same, except for 'inessential differences'. This leads to the obvious question: what counts as an
inessential difference?" (Sec. 1, p1)

**Relevance to our paper.** Constructive abstraction here is the notion that Geiger et al. 2025 and Sutter et al. 2025
build on, and that our featurization inherits through Geiger. The paper is the nearest formal discussion of when two
models count as "the same up to inessential differences". That is exactly the question our kept clause answers for two
networks under one featurization: a positive rescaling of the mechanism is inessential and anything else is not. The
paper leaves "inessential" to "the eye of the beholder", which helps frame our choice as a choice. That matters because
the draft's open question is whether "kept up to a gain" is too strict (trained fine-tunes turn writes rather than scale
them). It does not discuss neural networks or fine-tuning. Optional cite, beside Geiger 2025, where the kept clause is
motivated.

**For the digest.**
- `3-definition`: the formal source of constructive abstraction. It poses "what counts as an inessential difference" between two causal models, which our kept clause (equal up to a positive scalar) answers for two networks.
