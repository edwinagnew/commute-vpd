---
slug: shone2026-neuron-basis-theory
title: "Returning to the Neuron Basis for Interpretability, Part 1: Answering Three Objections"
authors: Shone & Dunlap
year: 2026
id: https://alignment.anthropic.com/2026/neuron-basis-theory/
tags: [BASELINE, CITE-FIX, BACKGROUND]
primary_tag: CITE-FIX
in_our_bib: no
cited_in_draft: no
themes: [baselines, other]
read_from: Anthropic blog, main text and Limitations verbatim (an Opus web-fetch agent, 2026-11-25); appendix summarised only
digest_buckets: []
importance: 5
---

# Returning to the Neuron Basis, Part 1: three objections (Shone 2026)

**What it does.** It argues that MLP neurons are a usable interpretable basis. Each sign of a neuron adds one main concept, and neurons that add two unrelated concepts are about 1% or fewer. Neurons write in a roughly fixed basis across layers, and facts are distributed across neurons but still interpretable. Models: Llama-3.1-8B, Qwen3-8B and 14B, OLMo-3-32B.

**Relevance to our paper.** Cite it beside cammarata2026clean and goodfire2026neurons at sec4_setup.tex:15, for the point that a neuron can be treated as a feature. It does not support singling out strongly firing neurons or our 97th-percentile cut: it holds that a weak activation adds a smaller amount of the same concept. It cites Gao (2026), "Neural Networks Are Not Already Clean" (LessWrong; not checked here) as a counterpoint.
