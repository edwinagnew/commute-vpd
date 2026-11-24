---
slug: karan2026-base-sampling
title: "Reasoning with Sampling: Your Base Model is Smarter Than You Think"
authors: Aayush Karan et al.
year: 2026
id: 2510.14901
tags: [EH, SUPPORTS, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: arXiv HTML: abstract, introduction, method section 3 (power distribution, MCMC), main results table; identified from Oh et al. 2026's bibliography (ICLR 2026)
digest_buckets: [4-supports]
importance: 3
---

# Reasoning with Sampling: Your Base Model is Smarter Than You Think (Karan 2026)

**What it does.** Asks whether RL's single-shot reasoning gains can be had from the base model by sampling alone. It
samples from the power distribution p(x)^α of the base model with a Metropolis-Hastings sampler that resamples token
blocks by base-model likelihood. No training, data or verifier. Models: Qwen2.5-Math-7B, Qwen2.5-7B, Phi-3.5-mini;
compared against GRPO on MATH.

**Key result.** On Qwen2.5-Math-7B, power sampling reaches 0.748 on MATH500 against GRPO's 0.785 (base 0.496), and
beats GRPO out of domain on HumanEval (0.573 vs 0.537) and AlpacaEval 2.0 (2.88 vs 2.38). It avoids the pass@k
diversity collapse of RL.

**Quote.** "sampling directly from the base model can achieve single-shot reasoning capabilites on par with those from RL." (Section 1)

**Relevance to our paper.** Behavioural evidence for the sharpening form of elicitation on maths RL: most of GRPO's gain
is reachable by re-weighting the base model's own distribution. It belongs in the elicitation-versus-sharpening
paragraph the draft plans, and it is a useful foil for our Qwen RL results: a behaviour reachable by re-weighting base
outputs says nothing about whether the trained weights reuse base mechanisms. No model internals.

**For the digest.**
- 4-supports: behavioural support that maths RL mostly sharpens the base distribution; context for our Qwen RL arm, not a test of our definition.
