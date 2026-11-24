---
slug: chen2024-safety-neurons
title: "Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons"
authors: Jianhui Chen et al.
year: 2024
id: 2406.14144
tags: [BASELINE, DIFFING, FT-INTERP, KNOWN-ANSWER, ALIGNMENT]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes, baselines]
read_from: arXiv HTML (abstract, sections 1-4 in full, section 5 skimmed)
digest_buckets: [1-missing-method, 2-known-answer-case]
importance: 5
---

# Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons (Chen 2024)

**What it does.** Llama2-7B, Mistral-7B, Gemma-7B and Qwen2.5-3B are trained with SFT on ShareGPT, then DPO on HH-RLHF-Harmless. Both stages use (IA)³ on the MLPs only, which rescales neuron activations and leaves the neurons' key and value vectors unchanged. "Inference-time activation contrasting" ranks MLP neurons by the RMS difference of their activations between the SFT and DPO models on the same generated text. "Dynamic activation patching" copies the DPO model's activations of the top neurons into the SFT (or base) model at every generation step. The causal effect is the share of the DPO-vs-SFT safety-score gap recovered, scored by a cost model.

**Key result.** Patching about 5% of neurons recovers over 90% of DPO's safety on Beavertails from SFT (e.g. +101% Llama2, +96% Gemma). The effect transfers to RedTeam, HarmBench and JailBreakLLMs, with general ability nearly unchanged. Random neurons matched per layer do far worse, and the ablation-based baselines (Wanda pruning, SN-Tune) fail the same check. The neuron sets agree across five seeds (overlap > 0.95). Safety and helpfulness neurons overlap heavily but need different activation patterns, which the paper offers as an account of the alignment tax.

**Quote.** "after patching approximately 5% of all the neurons, SFT can recover over 90% of DPO's safety performance" (Section 4.2)

**Relevance to our paper.** Two uses. (1) Method: activation contrasting plus dynamic patching is a cross-model patching locator for "what the fine-tune changed". It would run on our 67M model and on Qwen3-1.7B, and it is the closest neuron-level relative of our neuron CMAP baseline. Note that it patches values across models, which our definition avoids (we compare mechanisms, not values). (2) Known answer by construction: their fine-tunes are (IA)³-only, so every MLP neuron's mechanism is the base one times a gain, which is reuse under our kept clause at the neuron featurization. It is a realistic-scale DPO safety fine-tune where "rescaled existing neurons" is guaranteed and the behaviour is shown to run through about 5% of them. That makes it a natural positive control for our gain-only (IA³-style) refits and our ρ at neurons. It says nothing about full fine-tunes, where rotation is possible.

**For the digest.**
- 1-missing-method: activation-contrast ranking plus dynamic cross-model activation patching (gap recovered) is a published fine-tune locator we could run as a neuron baseline beside CMAP.
- 2-known-answer-case: DPO safety fine-tunes of four 3-7B models with (IA)³ on the MLPs only, so the update is a pure neuron rescale (reuse by construction) and the behaviour is causally localised to about 5% of neurons.
