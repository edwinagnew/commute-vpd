---
slug: mueller2025-right-mediator
title: "The Quest for the Right Mediator: Surveying Mechanistic Interpretability Through the Lens of Causal Mediation Analysis"
authors: Aaron Mueller et al.
year: 2025
id: 2408.01416
tags: [DEFINITION, CAUSAL, BACKGROUND]
primary_tag: DEFINITION
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, reuse-definitions]
read_from: "PDF text (v3, Computational Linguistics), local copy: pp1-8 (framing, the definition of mechanism, indirect effect, the four criteria), pp26-28 (recommendations, including footnote 28 on fine-tuning); pp9-25 (mediator types, search methods) skimmed by grep"
digest_buckets: [3-definition]
importance: 3
---

# The Quest for the Right Mediator: Surveying Mechanistic Interpretability Through the Lens of Causal Mediation Analysis (Mueller 2025)

**What it does.** A survey that recasts mechanistic interpretability as causal mediation analysis. Nodes of the network
(neurons, heads, layers, non-basis-aligned directions, SAE features) are candidate mediators between input and output.
Methods are sorted by mediator type and by search method (exhaustive, gradient-based attribution, alignment search). It
proposes four criteria for a good mediator: sparsity, generality, selectivity and faithfulness.

**Key result.** No new experiments. Its thesis is that the right mediator depends on the analyst's goal. It recommends
SAEs and other unsupervised methods for open-ended explanation and alignment search (DAS) for testing a stated hypothesis.
It warns that optimised alignment modules can "directly learn the phenomenon, rather than extracting it from the model"
(citing Sutter et al.).

**Quote.** "In the causality literature, a mechanism is defined as a causal chain from cause X to effect Y (Salmon 1984;
Pearl 2009). The mechanistic interpretability literature, while closely related to causal interpretability, does not
enforce this causally grounded definition of mechanism" (Sec. 2, p5)

**Relevance to our paper.** It is a citable statement that the field lacks an agreed definition of mechanism, which is
our motivation for one ("there is no consensus in the interpretability literature over what constitutes a mechanism").
Its causal-chain sense of "mechanism" differs from ours. We use Geiger's sense: the function that computes a variable
from its parents. The draft could say so in one line. The survey says of non-basis directions that "one may need
to rediscover these directions if model fine-tuning, adaptation, or editing is part of the study" (p27). Its footnote 28
adds that Prakash et al. found the same components before and after fine-tuning. Our design applies one base featurization to F unchanged, a deliberate choice against that advice,
and the draft should say why: fixing τ is what makes the comparison of mechanisms possible. The survey does not define
reuse or compare two models.

**For the digest.**
- `3-definition`: a survey-level definition of mechanism as a causal chain, and the claim that the field enforces none. It is a citation for our "no consensus" sentence and a contrast with Geiger's per-variable mechanism, which we use.
