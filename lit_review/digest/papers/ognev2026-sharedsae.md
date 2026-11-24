---
slug: ognev2026-sharedsae
title: "SharedSAE: One Feature Dictionary Across Language Models"
authors: Ognev et al.
year: 2026
id: 2609.04344
tags: [DIFFING, SUPPORTS]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: "arXiv HTML: abstract, introduction, related work, Section 5 (adaptation, Table 4)"
digest_buckets: [4-supports]
importance: 3
---

# SharedSAE: One Feature Dictionary Across Language Models (Ognev 2026)

**What it does.** Trains one sparse dictionary shared by several language models (Qwen3, Llama-3.2, OLMo-2, Gemma-3 at
~1B), with per-model encoder and decoder pairs, scale-normalised TopK selection and "model dropout" so each model can
select latents alone. A new model is attached afterwards by training only its encoder, decoder and biases against the
frozen dictionary.

**Key result.** The shared dictionary keeps 96.6% of dedicated SAEs' mean explained variance, and its latents
correlate across models 1.8x as strongly as post-hoc-matched separate SAEs. Held-out models attach at about 0.97-0.98
of their dedicated SAE's explained variance. Llama-3.2-1B-Instruct, whose base sibling is in the seed, attaches at
1.006 of its own dedicated SAE.

**Quote.** "so the labels learned on base models carry over to this instruct model without loss of reconstruction." (Section 5)

**Relevance to our paper.** A further instance, alongside Kissane et al., of a base-model featurization reconstructing
a post-trained model as well as a dedicated one, which supports our choice to fix the featurization from M and apply
it unchanged to F. It reads activations only, says nothing about whether mechanisms are kept, and the instruct result
is a single model. Its frozen-dictionary attachment could in principle serve as a diffing tool (which latents the
instruct model selects differently), but the paper does not do this, so it is not a baseline we need.

**For the digest.**
- 4-supports: a base-trained dictionary reconstructs an instruct model at 1.006 of a dedicated SAE, consistent with fixing τ from the base model.
