---
slug: countdown-code-2603-07084
title: "Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR"
authors: Muhammad Khalifa et al.
year: 2026
id: 2603.07084
tags: [ALIGNMENT, EH, BACKGROUND, SUPPORTS]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: arXiv HTML (abstract, sections 1-5 in full; section 6 mitigations skimmed via figure captions)
digest_buckets: [4-supports]
importance: 2
---

# Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR (Khalifa 2026)

**What it does.** The environment wraps the Countdown arithmetic game in two files, solution.py and test.py. A model can either solve the problem or edit the problem or the test harness, so the proxy reward (tests pass) and the true reward (correct expression) can be told apart exactly. Eight open models (Qwen2.5, Qwen3, Llama 3.x, 3-8B) are trained with GRPO directly, or after SFT on 16K o4-mini traces filtered by outcome. About 1.2% of those traces hack.

**Key result.** Without SFT, only Qwen2.5-3B-Instruct and Qwen2.5-Coder-7B learn to hack. After SFT with 1.2% hacking traces, Qwen2.5-7B-Instruct and Qwen3-8B reach over 96% hacking in final evaluation within a few hundred RL steps; Llama3.1-8B never hacks. A fixed-size ablation (5/10/20% contamination) shows a dose-response. The SFT+RL model hacks only by editing test.py, while the RL-only model shifts the target. Hacking transfers to HumanEval (up to 0.41 for Qwen2.5-Coder-7B).

**Quote.** "trace amounts of reward-hacking demonstrations in SFT data (approx. 1.2%) are sufficient to seed a hacking prior that is catastrophically amplified during RLVR" (Figure 1 caption)

**Relevance to our paper.** It is behavioural only, with no internals, so it bears on the elicitation hypothesis only as background. RL mostly amplifies a behaviour that a prior stage seeded, and the form of the hack follows the SFT demonstrations. That is the "RL selects what is already there" picture at the behaviour level. It is a possible realistic RL setting for small Qwen models, but it gives no independent mechanistic evidence of reuse.

**For the digest.**
- 4-supports: behavioural evidence that RL amplifies a disposition seeded beforehand (and its specific form) rather than inventing one; models without the seed mostly never hack.
