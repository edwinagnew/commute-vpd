---
slug: vig2020-causal-mediation
title: "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"
authors: Vig et al.
year: 2020
id: 2004.12265
tags: [CAUSAL, BACKGROUND, CITE-FIX]
primary_tag: CAUSAL
in_our_bib: vig2020
cited_in_draft: "yes, but only in struck-out (old) text in main.tex and sec4_setup.tex; the new text cites prakash2024, so once changes are accepted vig2020 is cited nowhere"
themes: [causal-abstraction, baselines]
read_from: "arXiv PDF 2004.12265v2 (the expanded version, retitled 'Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias') via fleet/pdf-text: abstract, Sec. 1-3 (total, natural direct and indirect effects), Sec. 5 (results on heads and neurons); appendices not read"
digest_buckets: []
importance: 3
---

# Investigating Gender Bias in Language Models Using Causal Mediation Analysis (Vig 2020)

**What it does.** Brings Pearl's causal mediation analysis to transformer interpretation. An intervention on the input
(e.g. "The nurse said that" with nurse changed to man) has a total effect on the relative probability of "he" vs
"she". The natural indirect effect of a mediator (a neuron or an attention head) is read by setting the mediator to
the value it takes under the intervention while the input stays as is; the direct effect holds it at its original
value. Applied to GPT-2 sizes (and a randomly initialized control) on professions templates, Winobias and Winogender.

**Key result.** Bias effects are sparse, concentrated in a small part of the network: on Winobias in GPT-2 small, a
greedy choice of 10 heads, mostly in layers 4-5, matches the indirect effect of intervening on all 144 heads at once.
For neurons, the strongest effects are in the embeddings and the first layer. Joint and summed head effects differ by
only 0.7% to 11.3%. Larger models show larger total effects.

**Quote.** "gender bias effects are (i) sparse, concentrated in a small part of the network; (ii) synergistic,
amplified or repressed by different components; and (iii) decomposable into effects flowing directly from the input
and indirectly through the mediators." (Abstract)

**Relevance to our paper.** The standard origin citation for activation patching on neurons and heads (indirect
effect = patch a component's value from a counterfactual input). It is within one model, between two inputs; it does
not compare a base model with a fine-tune, so it is not the source of our cross-model neuron share. The current draft
rightly moves the neuron share's citation to Prakash et al.'s CMAP, but that leaves vig2020 cited only in struck-out
text. If we keep it, the natural place is the "needed" clause or the methods, as the origin of single-component
interchange in language models alongside Geiger et al.; otherwise drop it from refs.bib. Minor: the arXiv v2 title
differs from the bib title (the bib uses the NeurIPS 2020 title; same arXiv ID).

**For the digest.**
- (no bucket; CITE-FIX note: only cited in \old{} text, so it will be orphaned when changes are accepted)
