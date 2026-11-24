---
slug: gurnee2024-universal-neurons
title: "Universal Neurons in GPT2 Language Models"
authors: Gurnee et al.
year: 2024
id: 2401.12181
tags: [BACKGROUND, CAUSAL]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: abstract, intro, section 4.1 method (excess correlation), section 5 overview, appendix on universal-neuron counts, conclusion and limitations"
digest_buckets: []
importance: 2
---

# Universal Neurons in GPT2 Language Models (Gurnee 2024)

**What it does.** Measures whether individual neurons recur across five GPT-2 small and medium models trained from
different seeds (with Pythia-160M checks). For every neuron pair across models, activation correlation is computed over
100M tokens; a neuron's "excess correlation" is its mean maximum correlation across models minus the same quantity in a
randomly rotated basis. Universal neurons (excess correlation > 0.5) are then characterised by weight statistics,
interpreted, and studied in simple circuits (prediction/suppression of token sets, entropy control via the final layer
norm, switching attention heads off).

**Key result.** Only 1-5% of neurons are universal (1.23% GPT-2 medium, 1.26% Pythia-160M, 4.16% GPT-2 small-a). They
are usually interpretable and fall into a few families; late-layer prediction neurons precede suppression neurons.

**Quote.** "only about 1-5% of neurons are universal across models, constituting another piece of evidence that
individual neurons are not the appropriate unit of analysis for most network behaviours" (Conclusion)

**Relevance to our paper.** It compares models trained from scratch, not a base and its fine-tune, so it bears on us
only through the neuron featurization: its rotated-basis baseline is a good template for a control on our neuron and
top-activating-neuron readings, and its conclusion that neurons are rarely the right unit sits beside our finding that
at neurons almost everything reads as reuse (≥ 0.999). Of little direct use otherwise.

**For the digest.**
- (no bucket)
