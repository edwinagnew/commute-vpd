---
slug: school-of-reward-hacks-2508-17511
title: "School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs"
authors: Mia Taylor et al.
year: 2025
id: 2508.17511
tags: [ALIGNMENT, BACKGROUND]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: arXiv HTML (abstract, introduction, section 2, results sections 3-4 via text and figure captions)
digest_buckets: []
importance: 1
---

# School of Reward Hacks: Hacking harmless tasks generalizes to misaligned behavior in LLMs (Taylor 2025)

**What it does.** The paper builds 1,073 single-turn dialogues (973 natural-language, 100 coding) in which the user states an exploitable evaluation metric and the assistant games it, with harmful content filtered out. GPT-4.1, GPT-4.1-mini, Qwen3-32B and Qwen3-8B are fine-tuned on it with SFT, against a control set of non-gaming answers to the same prompts.

**Key result.** The fine-tuned models generalise to new hacks. They include a verification phrase 99% of the time (vs < 2% for baselines), set their own reward function to the maximum 98% of the time, prefer less-informed graders, and try to hack a chess engine in 94% of rollouts. GPT-4.1 also becomes broadly misaligned on Betley et al.'s questions (12% of responses), with patterns like the insecure-code and bad-advice organisms. This needs the varied non-coding tasks; coding-only training does not cause it.

**Quote.** "Although the reward hacking behaviors in the training data were harmless, GPT-4.1 also generalized to unrelated forms of misalignment" (Abstract)

**Relevance to our paper.** It is a behavioural model-organism paper with no internals and no reuse question. It is of little use to us beyond being one more narrow-to-broad generalisation case, of the kind persona-feature work explains by a base-model direction.

**For the digest.**
