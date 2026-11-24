---
slug: nanda2023-attribution-patching
title: "Attribution Patching: Activation Patching At Industrial Scale"
authors: Nanda
year: 2023
id: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
tags: [CAUSAL, BASELINE]
primary_tag: BASELINE
in_our_bib: nanda2023attribution
cited_in_draft: yes
themes: [baselines, causal-abstraction]
read_from: "blog post full text (fetched with curl, HTML stripped): TLDR, method, 'Does this work in practice', theory of where the linear approximation fails, variants, conclusion; the accompanying notebook not run or read"
digest_buckets: []
importance: 3
---

# Attribution Patching: Activation Patching At Industrial Scale (Nanda 2023)

**What it does.** Introduces attribution patching: approximate the effect of patching each activation from a clean run
into a corrupted run by a first-order Taylor term, (clean_x - corrupted_x) times the gradient of the patch metric at
the corrupted run, so every patch comes from two forward and one backward pass. Tested against real activation
patching on the IOI circuit in GPT-2 small, at the level of residual streams, layer outputs, heads, and queries, keys,
values and attention patterns. Write-up of a 2022 Anthropic project.

**Key result.** No single number; scatter plots of attribution vs activation patching. The approximation is "bad" for
"big" activations (residual stream, and MLP0, which acts as an extended embedding in GPT-2 small) and "good" for other
layer outputs, head outputs and sub-head activations, recovering the outline of the IOI circuit. It is expected to fail
when a patch moves a non-linearity between saturated and unsaturated regions (a confident log-prob, an attention
weight near 0 or 1, a GELU's negative tail), and it misses effects that need several variables at once.

**Quote.** "it's a bad approximation for "big" activations like residual stream patching and for MLP0, a good
approximation for all other layer outputs, head outputs, and more zoomed in head activations" (Does This Work In
Practice?)

**Relevance to our paper.** The origin citation for the attribution score that orders neurons in our neuron-level
CMAP baseline; we cite it correctly. Two of its caveats apply directly to our setting: our 67M model has only four
blocks, so MLP0 neurons are a large share of the ranked neurons, and the post says the approximation is worst there;
and patching a fine-tune's firing into the base is a large patch, which is where saturation breaks the linear
estimate. We score real patches of prefixes and use attribution only for the order, so these caveats cost at most a
worse order. But a worse order inflates the neuron count noisily, which could itself produce some of the 3 to 5
reversed pairs we report for neuron CMAP; a check with an activation-patching order on one pair, or one clause naming
the caveat, would keep that failure from being blamed on the approximation.

**For the digest.**
- (no bucket; already cited correctly; its MLP0 and saturation caveats apply to our neuron order)
