---
slug: du2026-fixed-sae-track
title: "What Does an LLM Learn from Reinforcement Learning? A Mechanistic Interpretability Perspective with Fixed-SAE Track"
authors: Du, Tang et al.
year: 2026
id: 2609.15064
tags: [FT-INTERP, EH, DYNAMICS, BASELINE, KNOWN-ANSWER, SUPPORTS]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, elicitation, training-dynamics]
read_from: "arXiv HTML (v1): abstract, intro, Sec. 3 method, Sec. 4.1-4.4 results, conclusion, appendix headings; the synthetic known-feature benchmark promised in the abstract has no results in the v1 body I could find"
digest_buckets: [1-missing-method, 2-known-answer-case, 4-supports]
importance: 5
---

# What Does an LLM Learn from RL? Fixed-SAE Track (Du 2026)

**What it does.** It trains one TopK SAE on late-layer residual activations pooled across a base model and all its GRPO
checkpoints, so each latent is one fixed direction across training. It tracks each latent's mean activation (drift),
reads what it encodes, steers the base model with the amplified latents' decoder directions, and ablates latents in the
RL model. The main setting is Qwen2.5-1.5B-Instruct on maths, layer 27, 13 checkpoints.

**Key result.** Drift is small and gradual, and grows with depth (max drift 0.013 at layer 6, 0.060 at layer 27). It sits
in about ten latents: amplified ones encode formatting scaffolds (operators, LaTeX edges, step breaks), and suppressed
ones encode problem content. Adding the amplified directions to the base model raises strict accuracy from 0.050 to
0.192, against 0.175 for the RL model, and random directions do not. Longer training adds a real reasoning gain
(lenient accuracy 0.300 to 0.408). Ablating an arithmetic latent set hurts accuracy, but injecting it into the base model
recovers little.

**Quote.** "short RL mainly installs and stabilizes scaffolds that elicit latent capability, while sufficiently sustained RL and models with sufficient headroom can additionally acquire modest genuine reasoning improvements." (Sec. 4.4)

**Relevance to our paper.** It is close in design to our SAE featurization (one dictionary held fixed, both models read
through it). Unlike ours, the dictionary is pooled across checkpoints rather than fixed from the base model, and it
compares by activation value, not by mechanism. Its steering-recovers-the-gain test, with a random-direction control, is
the same logic as our mean-difference steering share, so it adds support that steering is a fair cheap reading. Its
short-RL format gain is a candidate realistic known answer: independent evidence (steering the base with base-readable
directions reproduces the gain) that the format change runs through what the base already has. The "sustained RL adds
reasoning" phase is a weaker candidate for new. The synthetic benchmark with features known by construction would be the
closest comparison to our built edits, but v1 reports no results for it.

**For the digest.**
- 1-missing-method: pooled fixed-SAE drift plus steering; near our SAE and steering readings; no new tool to add, but cite as parallel work.
- 2-known-answer-case: Qwen2.5-1.5B-Instruct, GRPO on maths: base steering on amplified latents recovers the strict-accuracy gain (0.05 to 0.19 vs RL 0.175) = format reuse, by value-level evidence.
- 4-supports: short RL is mostly elicitation; sustained RL adds a modest real gain.
