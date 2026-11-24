---
slug: merullo2023-circuit-component-reuse
title: "Circuit Component Reuse Across Tasks in Transformer Language Models"
authors: Jack Merullo et al. (Eickhoff, Pavlick)
year: 2023
id: 2310.08744
tags: [CAUSAL, DEFINITION, EH, BACKGROUND]
primary_tag: CAUSAL
in_our_bib: no
cited_in_draft: no
themes: [reuse-definitions, causal-abstraction, elicitation]
read_from: "arXiv HTML v3: abstract, Sec. 1, 4 (Colored Objects circuit), 4.5, 5 (intervention), 6, 7"
digest_buckets: [3-definition]
importance: 4
---

# Circuit Component Reuse Across Tasks in Transformer Language Models (Merullo 2023)

**What it does.** Reproduces the IOI circuit (Wang et al. 2022) on GPT2-Medium by path patching, then finds the circuit for
a different task, Colored Objects ("What color is the pencil?"), in the same model. It compares the heads and their roles,
then intervenes by hand on the attention patterns of three inhibition heads and one negative mover head, forcing them to
attend to the wrong colours as the IOI circuit would. It then checks the downstream effect on the mover heads.

**Key result.** 25 of the 32 most important heads (78%) are shared between the two circuits. The difference is three
inhibition heads and a negative mover, swapped for three content-gatherer heads. The intervention raises Colored Objects
accuracy from 49.6% to 93.7% with zero new mistakes (78.1% with the negative mover alone, 81.5% with the inhibition heads
alone). Mover heads attend 8.7% less to wrong colours, and their logit attribution rises about 3x.

**Quote.** "the inhibition-mover subcircuit is itself a manipulable structure within the model that is invariant to the
highly different input domains that we used in our experiments." (Fig. 5 caption)

**Relevance to our paper.** It shows reuse *across tasks within one model*, not across a fine-tune, so it is not a
known-answer case for us. Its notion of reuse is functional: the same heads in the same algorithmic roles, found by path
patching. That is worth citing as the circuit-level antecedent of "reuse", and contrasting with ours, which compares one
variable's mechanism between M and F. The intervention is also a hand-made "elicitation": the base model has the IOI
subcircuit but does not engage it on Colored Objects, and switching it on recovers the task. A Colored Objects fine-tune
of GPT2-Medium therefore has a predicted reuse answer (engage the inhibition heads). That would make a strong known-answer
candidate if someone fine-tuned and checked, but the paper does not.

**For the digest.**
- `3-definition`: a functional, cross-task notion of component reuse (same heads, same roles). It is a precedent to contrast with our between-model kept clause; the paper's intervention suggests a predicted-reuse fine-tune (GPT2-Medium on Colored Objects) that nobody has run.
