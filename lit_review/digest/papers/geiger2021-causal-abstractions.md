---
slug: geiger2021-causal-abstractions
title: Causal Abstractions of Neural Networks
authors: Atticus Geiger, Hanson Lu, Thomas Icard, Christopher Potts
year: 2021
id: 2106.02997
tags: [CAUSAL, DEFINITION, CITE-FIX]
primary_tag: CAUSAL
in_our_bib: no
cited_in_draft: no
themes: [causal-abstraction, reuse-definitions]
read_from: "PDF pages 1-9 (abstract, method, MQNLI results, comparison with probes)"
digest_buckets: [3-definition]
importance: 4
---

# Causal Abstractions of Neural Networks (Geiger 2021)

**What it does.** Introduces causal abstraction analysis for neural networks (NeurIPS 2021). Representations are aligned
with the variables of an interpretable high-level causal model. Interchange interventions then test whether the aligned
representations play the variables' causal role: a representation computed on one input is swapped into a run on another,
and the output is checked against the high-level model's counterfactual. The case study is natural language inference on
MQNLI, comparing a fine-tuned BERT classifier with a BiLSTM against a natural-logic causal model.

**Key result.** BERT reaches 88% on the hard generalization split, against 46% for the BiLSTM, and 49% for BERT without
pretraining. Over 1,000 examples, the largest set on which the interchange interventions all agree with the natural-logic
model is large for BERT and much smaller for the BiLSTM. The paper concludes that BERT partly realises the natural-logic
causal structure. Probes, by contrast, are argued to give no causal characterisation.

**Quote.** "an interchange intervention, in which a neural representation created during prediction on a “base” input is interchanged with the representation created for a “source” input" (Section 3)

**Relevance to our paper.** It is the origin of the interchange intervention our Def. 2 uses to decide which variables are
"needed". The draft cites only the 2025 foundation paper (`geiger2025`). Citing this one where interchange interventions are
introduced is cheap and conventional, and can be optional. It does not compare two networks' mechanisms, so it adds nothing
to the kept clause. Its "largest clique of inputs on which abstraction holds" is one more approximate-abstraction measure
our ρ could be contrasted with.

**For the digest.**
- `3-definition`: original interchange-intervention definition of when a network realises a causal variable. Cite it at Def. 2 beside Geiger 2025 (CITE-FIX: not in our bib).
