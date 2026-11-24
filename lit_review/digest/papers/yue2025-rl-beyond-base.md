---
slug: yue2025-rl-beyond-base
title: "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
authors: Yue et al.
year: 2025
id: 2504.13837
tags: [EH, SUPPORTS, BACKGROUND]
primary_tag: EH
in_our_bib: yue2025
cited_in_draft: yes
themes: [elicitation]
read_from: "arXiv HTML (v5): abstract, intro, pass@k method, results on maths/code/vision, perplexity and coverage analysis, distillation comparison, limitations"
digest_buckets: [4-supports]
importance: 5
---

# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (Yue 2025)

**What it does.** It asks whether RL with verifiable rewards (RLVR) gives a model reasoning the base model lacks, using
pass@k at large k (up to 1024) as the measure of a model's reasoning boundary. It compares base and RLVR models (Qwen2.5
7B-32B, LLaMA, DeepCoder, vision models) on maths, code and visual reasoning, checks correct CoTs by hand, and measures
the base model's perplexity on RL models' outputs. It contrasts RL with distillation from a stronger teacher.

**Key result.** RL models win at small k, but base models catch up and overtake them at large k. Problems solved after
RL are "nearly a subset" of those the base model solves, and the base model assigns low perplexity to RL outputs.
Distilled models, in contrast, go beyond the base model's boundary. The gap between RL pass@1 and base pass@256 stays
above 40 points across six RL algorithms.

**Quote.** "all the reasoning paths exploited by the RLVR model are already present in the sampling distribution of the base model." (Sec. 4 summary)

**Relevance to our paper.** The draft cites it correctly, as behavioural evidence for elicitation from RL. It is exactly
the kind of evidence our gap sentence calls behavioural: it says nothing about whether the same mechanism computes the
answer, and our definition can tell two fine-tunes with one behaviour apart. Its RL-versus-distillation contrast is a
behavioural known answer we could borrow: if our Qwen RL and SFT/distillation runs were scored, the prediction would be
that distillation reads more new. The draft's Qwen results (no call; ρ does not separate a learning RL run from a
collapsed one) cannot yet test that.

**For the digest.**
- 4-supports: behavioural support for EH under RL; the distillation contrast is a behavioural prediction our Qwen line could test.
