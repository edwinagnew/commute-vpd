---
slug: geiger2024-finding-alignments
title: "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"
authors: Atticus Geiger et al.
year: 2024
id: 2303.02536
tags: [CAUSAL, DEFINITION, CITE-FIX]
primary_tag: CAUSAL
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, reuse-definitions]
read_from: "PDF text (v4, CLeaR 2024), local copy: abstract, intro, Defs 1-5 (pp1-8), hierarchical-equality and random-network results (pp9-11), MoNLI BERT results (pp11-13), FAQ appendix (pp22-23)"
digest_buckets: [3-definition, 5-cite-fix]
importance: 4
---

# Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations (Geiger 2024)

**What it does.** Introduces distributed alignment search (DAS). Instead of brute-force search over sets of neurons, it
learns an orthogonal rotation of a layer by gradient descent, so that subspaces of the rotated layer behave, under
interchange interventions, like the variables of a chosen high-level algorithm. The network and the algorithm are frozen;
only the rotation is trained. It is tested on an MLP trained on hierarchical equality and on BERT fine-tuned on MoNLI
(monotonicity NLI). The paper also defines the graded score it uses, interchange intervention accuracy (IIA).

**Key result.** DAS finds alignments that localist search misses. On hierarchical equality, the "both equality relations"
algorithm reaches perfect or near-perfect IIA under DAS, against 0.60 for the best brute-force alignment and 0.73 for the
best localist one. In fine-tuned BERT, layer 9 at intervention size 256 gives IIA 1.00 for the negation and
lexical-entailment algorithm. As a check, DAS on a frozen random MLP stays at chance (0.50) for small hidden sizes and
climbs only to 0.64 when the hidden size is 4096 against a 16-dimensional input. The authors read this as "DAS cannot
construct entirely new behaviors from random structure". Sutter et al. 2025 later contest this with more expressive maps.

**Quote.** "What this reveals is that there is an alignment, but not in the basis we chose. Since the choice of basis
was arbitrary, our negative conclusion about the causal abstraction relation was spurious." (Sec. 2, p5)

**Relevance to our paper.** This is the standard source for the distributed interchange intervention and for IIA. Our
"needed" test (Def. 2) is the same interchange idea, but at a featurization fixed from the base model, not one learned per
algorithm. Their quote about basis choice is the clearest precedent for our point that reuse is relative to the
featurization (at neurons almost everything reads as reuse). It also sets up the Sutter et al. objection, which the draft
answers by fixing the featurization. The paper analyses a fine-tuned BERT, but only one model per analysis: it never
compares a model with its base, so it is not a known-answer case or a diffing baseline. Their random-network control is
the analogue of our random-update floor. The draft cites Geiger 2025 for the interchange intervention; citing DAS beside
it, where we contrast a fixed featurization with a learned alignment, would be accurate.

**For the digest.**
- `3-definition`: the source of the learned-alignment interchange and IIA. Our needed set uses the same interchange idea at a featurization fixed from the base model, and the draft could cite this paper where it draws that contrast.
