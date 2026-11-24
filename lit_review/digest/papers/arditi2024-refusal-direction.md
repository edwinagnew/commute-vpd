---
slug: arditi2024-refusal-direction
title: "Refusal in Language Models Is Mediated by a Single Direction"
authors: Arditi et al.
year: 2024
id: 2406.11717
tags: [KNOWN-ANSWER, FT-INTERP, BASELINE, EH, ALIGNMENT, CITE-FIX]
primary_tag: KNOWN-ANSWER
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, elicitation, baselines]
read_from: "arXiv HTML: abstract, intro, sections 2-4 headings and text, App. J (base models) in full"
digest_buckets: [2-known-answer-case, 1-missing-method, 4-supports]
importance: 6
---

# Refusal in Language Models Is Mediated by a Single Direction (Arditi 2024)

**What it does.** For 13 open chat models up to 72B, takes the difference in means of residual activations between
harmful and harmless instructions and tests that one direction causally: ablating it from the residual stream, or
orthogonalising the weights against it (a rank-one weight edit), and adding it. It also attributes the direction to
components with direct feature attribution to explain adversarial suffixes, and in Appendix J checks whether the
direction already exists in the matching base models.

**Key result.** Ablating the direction removes refusal on harmful prompts with little effect on coherence, and adding
it makes models refuse harmless ones; weight orthogonalisation is a competitive white-box jailbreak. In the base models
(e.g. Llama-3 8B vs Instruct), harmful prompts already have high cosine similarity with the chat model's refusal
direction and harmless ones low, so the authors read the feature as pre-existing and repurposed.

**Quote.** "rather than developing the “refusal direction” from scratch during fine-tuning, this representation exists
already in the base model, and is repurposed for refusal during safety fine-tuning." (App. J)

**Relevance to our paper.** This is one of the most-cited realistic cases of "post-training hooks into a base-model
variable", and the draft does not cite it. Its evidence is exactly the kind our gap sentence describes: it reads the
direction's value in both models (cosine of activations), not its mechanism, so under our definition it shows the
variable exists in M, not that F computes it with M's mechanism or that the refusal behaviour's needed variables are
kept. That makes it a good contrast case to cite when we say reuse is compared "by mechanism, not by value", and a
candidate known-answer pair on Qwen3 (base vs instruct refusal) where the variable's reuse is widely believed. The
difference-in-means direction and its ablation are what our mean-difference steering share already uses, so no new
baseline is needed, but the rank-one weight orthogonalisation is a weight-space form of the same test.

**For the digest.**
- 2-known-answer-case: four base/chat pairs (e.g. Llama-3 8B vs Instruct); refusal direction present in base by activation cosine (value only, no mechanism test).
- 1-missing-method: direction ablation / weight orthogonalisation; covered by our mean-difference steering share.
- 4-supports: realistic post-training repurposes a pre-existing variable (EH at the level of variables, not mechanisms).
