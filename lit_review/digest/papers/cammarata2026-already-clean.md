---
slug: cammarata2026-already-clean
title: "Neural Networks Are Already Clean (Goodfire essay)"
authors: Nick Cammarata (with Curt Tigges)
year: 2026
id: https://already-clean.goodfire.com
tags: [BACKGROUND, BASELINE]
primary_tag: BACKGROUND
in_our_bib: cammarata2026clean
cited_in_draft: yes
themes: [baselines, other]
read_from: "essay full text from the web page (prose extracted from HTML; interactive figures not seen)"
digest_buckets: []
importance: 3
---

# Neural Networks Are Already Clean (Cammarata 2026)

**What it does.** An informal working-hypothesis essay arguing that polysemanticity in early model organisms came from
fixable problems (architecture, reward-hacked feature visualisation, ignoring negative SwiGLU poles, the double-negative
regime) rather than superposition. Evidence comes from vision models (InceptionV1 retrained, ConvNeXt, DINOv2) and
language models (Qwen, a neuron-only attribution graph of Kimi K3), and from SAEs trained on squeezed layers, whose
features decompose back into existing neurons.

**Key result.** Interpretable last-layer neurons go from ~5% (InceptionV1, old tools) to ~50% with better tools, ~70%
retrained and ~99% in a same-size ConvNeXt. About 90% of language-model neurons have "basically one job" plus related
offshoots ("recursive generalization"); weak activations can be ablated at little cost; removing eight neurons with a
Santa negative pole drops "Santa" from near-certain to 6%.

**Quote.** "The cause of polysemanticity in early model organisms wasn't superposition, but several ordinary, mostly
fixable issues." (subtitle)

**Relevance to our paper.** Cited with its companion article for treating strongly firing neurons as features, which is
an accurate use. It is an essay without a fine-tuning or diffing component, so it adds no baseline, known answer or
definition beyond the neuron featurization; its emphasis on negative poles is the same caveat noted for the companion
article (keep both tails on SwiGLU models). Little further use.

**For the digest.**
