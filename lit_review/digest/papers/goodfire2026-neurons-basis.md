---
slug: goodfire2026-neurons-basis
title: "Neurons as an Interpretable Basis"
authors: Curt Tigges et al. (Tigges, Kowal, McGrath, Lewis, Cammarata; Goodfire)
year: 2026
id: https://www.goodfire.com/research/neurons-as-an-interpretable-basis/article.md
tags: [BASELINE, SUPPORTS, BACKGROUND]
primary_tag: BASELINE
in_our_bib: goodfire2026neurons
cited_in_draft: yes
themes: [baselines, other]
read_from: "full-text markdown of the article (2026-11-10): abstract, introduction, definitions, SwiGLU background, sparsity section, conclusion, appendix on the recovery fine-tune; skimmed subcircuit/mode and evaluation sections"
digest_buckets: [4-supports]
importance: 4
---

# Neurons as an Interpretable Basis (Tigges et al., Goodfire 2026)

**What it does.** Argues that MLP neurons are usable units of interpretation without a learned dictionary. On
Qwen3.6-35B-A3B (MoE, SwiGLU, 5M neurons) it (1) ablates weak activations, keeping per layer only the top-320 of 4,096
routed neurons by absolute activation (or the p97/3 band), (2) traces each neuron-pole's subcircuits (inputs and
readers) and clusters them into "modes", and (3) has an agent write neuron descriptions scored by detection,
classification, simulation and input synthesis. Authors: Tigges, Kowal, McGrath, Lewis, Cammarata.

**Key result.** Ablating 92.2% of routed activations costs 1.2-2.2 points on log-likelihood benchmarks; reasoning
benchmarks loop, and a recovery fine-tune of ~100M tokens (SFT plus 9M tokens of RL) restores or exceeds them. Most
neurons have one mode or one dominant mode (>= 90% of causal effect). Descriptions detect firing 95% vs random text and
simulate p99.9+ firings well enough to recover 73% of ablation damage. Negative poles matter as much as positive ones.

**Quote.** "Our subsequent analyses of neurons in the fine-tuned model showed that the neurons retained their original
functions, remained interpretable, and had the same numbers of modes." (Sparsity section)

**Relevance to our paper.** We cite it, correctly, for the top-activating-neuron featurization and the top-neuron
share (Goodfire keep 7.8% of routed neurons; our sets are far sparser, as the draft says). Two points to check: its
unit is the neuron-pole and its band is p97/3, both tails, because SwiGLU neurons are bipolar; the draft keeps only the
upper tail on the 67M (a departure already stated in appG), but for Qwen3-1.7B (SwiGLU) the negative pole should be kept
or the departure stated. Its recovery fine-tune is described as correcting reasoning "rather than adding new
information", with neurons keeping their functions: an informal reuse claim on a real fine-tune, but backed only by
re-running the description pipeline, so not a known-answer case. It supports our finding that at neuron grain almost
everything reads as reuse, and gives no tool for diffing a fine-tune.

**For the digest.**
- 4-supports: neurons keep their functions and modes after a 100M-token recovery fine-tune (informal, description-level evidence), consistent with our neuron featurization reading nearly everything as reuse.
