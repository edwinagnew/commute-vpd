---
slug: ward2025-reasoning-ft-repurposes
title: Reasoning-Finetuning Repurposes Latent Representations in Base Models
authors: Ward et al.
year: 2025
id: 2507.12638
tags: [KNOWN-ANSWER, EH, FT-INTERP, DIFFING, DEFINITION, BASELINE, SUPPORTS, CITE-FIX]
primary_tag: KNOWN-ANSWER
in_our_bib: ward2025reasoning
cited_in_draft: yes
themes: [elicitation, interpreting-finetunes, reuse-definitions, baselines]
read_from: arXiv HTML full text (abstract, sections 1-5); appendices not read
digest_buckets: [2-known-answer-case, 3-definition, 4-supports, 1-missing-method, 5-cite-fix]
importance: 6
---

# Reasoning-Finetuning Repurposes Latent Representations in Base Models (Ward 2025)

**What it does.** Asks where the backtracking behaviour of a reasoning fine-tune comes from. The pair is Llama-3.1-8B
(base) and DeepSeek-R1-Distill-Llama-8B (SFT-distilled on R1 traces). The authors compute difference-of-means
"backtracking" steering vectors at layer 10, from tokens about 8 to 13 positions before a backtracking sentence, using
activations of either model on the same reasoning traces. They then steer each model with each vector. Logit lens and
probing check what the direction encodes. ICML 2025 Actionable Interpretability workshop paper, four pages.

**Key result.** A vector taken from the base model's activations induces backtracking in the reasoning model about as
well as one taken from the reasoning model. The two vectors have cosine about 0.74. Neither vector induces
backtracking in the base model. The direction does not project onto "wait"/"but" logits. It beats mean, noise,
self-amplification and other-category vectors. It is densely present in activations and does not cleanly predict
backtracking when used as a probe, so it is "one component" of the mechanism.

**Quote.** "while this representation is shared by both base and reasoning models, the representation only induces
backtracking in the reasoning model, suggesting that it has been repurposed as an input to the backtracking mechanism
during reasoning finetuning." (Sec. 1)

**Relevance to our paper.** This is the closest published case where a realistic post-training run is shown, by
intervention in both models, to build a behaviour on a base-model variable. It is also a clean example of the split our
definition draws. The upstream variable (the direction) is computed the same way in both models, so it reads as kept,
while the downstream mechanism that turns it into "Wait" is new: the base model ignores it. Under our Def. 3 the
behaviour would be partly reused and partly new, and the paper's own word "repurposes" names exactly that mixed case.
That makes it a good contrast for the definition section, a semi-known answer for the Qwen line (an RL or SFT
reasoning run where a base-derived steering vector works in F but not in M flags a new consumer), and support that
post-training at least partly elicits. Its method, steering F with a base-derived mean-difference vector and checking
that it fails in M, is cheap and runs on our setting. It is a two-sided variant of our mean-difference steering share.
The draft cites it only in the appendix list of baselines (main.tex, `sec:app_baseline`). It belongs in related work
on post-training as selection and on defining reuse.

**For the digest.**
- 2-known-answer-case: Llama-3.1-8B to R1-Distill; base-derived backtracking direction steers F (cos 0.74 to F's), not M: kept input, new consumer.
- 3-definition: "repurposed representation" = variable kept, downstream mechanism new; contrast with our per-variable kept/needed split.
- 4-supports: post-training reuses base representations rather than learning backtracking from scratch (partial, single model).
- 1-missing-method: two-sided steering test (M-derived vector in F vs in M) is a cheap reading we could add beside our steering share.
