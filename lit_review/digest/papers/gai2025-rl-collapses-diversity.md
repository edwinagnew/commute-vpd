---
slug: gai2025-rl-collapses-diversity
title: Differential Smoothing Mitigates Sharpening and Improves LLM Reasoning
authors: Jingchu Gai et al.
year: 2025
id: 2511.19942
tags: [EH, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: arXiv HTML: abstract, introduction, related work, section 3 setup and Propositions 3.1-3.2 (statements; inline math stripped)
digest_buckets: []
importance: 2
---

# Differential Smoothing Mitigates Sharpening and Improves LLM Reasoning (Gai 2025)

**What it does.** Gives a formal account of RL's diversity collapse in a simplified KL-regularised setting where RL
reweights trajectories sampled from the base policy, and derives a fix: penalise log-probability on correct
trajectories and add it on incorrect ones (DS-GRPO). Tested on 1.5B-8B models on Countdown and maths.

**Key result.** Two propositions: "selection bias" (a correct trajectory's chance of being reinforced rises with its base
probability) and "reinforcement bias" (its gain is proportional to its base probability). DS-GRPO improves both Pass@1
and Pass@k over vanilla GRPO and entropy heuristics, by up to 6.7% on AIME24.

**Quote.** "the process disproportionately reinforces correct trajectories that are already high-probability under the base LLM" (Section 3.2)

**Relevance to our paper.** Little. It is a behavioural, distribution-level account of why RL sharpens toward what the
base model already favours; at most a citation in the planned elicitation-versus-sharpening paragraph, and a reminder
that "high base probability" (our base prior) predicts what RL reinforces, which is the prior our key tasks pull apart
from the mechanism.

**For the digest.**
