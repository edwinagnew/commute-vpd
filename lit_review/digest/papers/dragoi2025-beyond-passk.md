---
slug: dragoi2025-beyond-passk
title: "Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries"
authors: Marius Dragoi et al.
year: 2025
id: 2510.08325
tags: [EH, DEFINITION, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "local PDF (arXiv v2, Oct 2025) pages 1-7: abstract, pass@k critique (Sec. 3), Cover@tau and its relation to pass@k (Secs. 4-5), experiments (Sec. 6, Table 1)"
digest_buckets: [3-definition]
importance: 2
---

# Beyond Pass@k: Breadth-Depth Metrics for Reasoning Boundaries (Dragoi 2025)

**What it does.** Argues that large-k pass@k, the usual evidence that base models have a wider "reasoning boundary" than
RLVR models, rewards lucky guesses on tasks with small answer spaces. Proposes Cover@tau, the fraction of problems solved
with per-sample success rate at least tau, and proves pass@k is a Beta(1,k)-weighted average of Cover@tau (so large k
looks only at tau near 0). Evaluates GRPO, PPO, GSPO, KL-Cov and an unlikeliness reward from Qwen2.5-7B-Instruct on
OMEGA (OOD splits) and Reasoning Gym.

**Key result.** On OMEGA's Probability task the base reaches pass@8192 = 1 because there are only 30 possible answers,
yet its Cover@tau collapses quickly. Averaged over five OMEGA OOD tasks, base cov@0.8 is 1.10 vs 11.98 (GRPO) and 23.34
(KL-Cov); rankings of RL methods change between pass@1 and Cover@tau.

**Quote.** "Pass@k at large k reflects the increasingly higher chance of success in the limit of the number of trials
rather than genuine reasoning, and can therefore be misleading." (abstract)

**Relevance to our paper.** A critique of the behavioural evidence (pass@k crossovers) that the elicitation debate
rests on: whether RL "expands the boundary" changes with the metric. That supports our introduction's point that
behavioural tests cannot settle EH and a mechanistic definition is needed. It offers a behavioural definition of a
capability boundary to contrast with ours, nothing more.

**For the digest.**
- 3-definition: a behavioural "reasoning boundary" (Cover@tau, reliability-thresholded) whose verdict on RL flips with the metric; contrast for our mechanistic definition.
