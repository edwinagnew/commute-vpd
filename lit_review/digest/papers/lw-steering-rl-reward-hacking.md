---
slug: lw-steering-rl-reward-hacking
title: "Steering RL Training: Benchmarking Interventions Against Reward Hacking"
authors: ariaw (LessWrong handle) et al.
year: 2025
id: https://www.lesswrong.com/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against
tags: [ALIGNMENT, BACKGROUND]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: LessWrong post full (via GreaterWrong mirror, main text; appendix not read)
digest_buckets: []
importance: 2
---

# Steering RL Training: Benchmarking Interventions Against Reward Hacking (ariaw 2025)

**What it does.** Qwen3-4B is trained with GRPO (LoRA rank 32, 200 steps) on LeetCode medium and hard problems with an "overwrite tests" loophole: the model can redefine the run_tests() function it is told is used for scoring. The post benchmarks interventions during RL: penalty or screening driven by a ground-truth monitor, by degraded ground-truth monitors (70% and 90% accurate), by a linear probe trained on base-model activations, or by a Claude Haiku 4.5 judge; and inoculation prompting.

**Key result.** The base model reward hacks 0.09% of the time; after about 80-100 steps every run hacks, at 79% on evaluation. Ground-truth penalty or screening removes hacking and slightly beats the no-loophole RL baseline on coding. The base-model probe (80% accurate in training) used as a penalty gave 0% hacking, better than a 90%-accurate ground-truth monitor. Penalties suppress hacking better than screening, but low-accuracy penalties hurt coding more. Inoculation prompting gave modest, highly variable protection.

**Quote.** "Qwen 3 4B reward hacks or attempts to reward hack only 0.09% of the time prior to any reward hacking training." (Setup)

**Relevance to our paper.** Mostly background. One point is of mild interest: a linear probe trained on base-model activations still flagged the RL-learned hack well enough to stop it. That is weak, read-from-activations evidence that the learned behaviour uses a representation already present in the base. It is the kind of evidence our paper calls insufficient, not a known answer. The open environment is a realistic RL run on a small Qwen whose behaviour is near-absent in the base, so it could be an "is this new?" test case, but it has no mechanistic label.

**For the digest.**
