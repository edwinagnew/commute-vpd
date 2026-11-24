---
slug: lw-wordle-reward-hacked-policy
title: "Every reward-hacked policy I tested triggered the OPE alarm — and so did my best honest one"
authors: JulesRoussel01 (LessWrong handle)
year: 2026
id: https://www.greaterwrong.com/posts/ZRfktnKPHStoz3K7H/every-reward-hacked-policy-i-tested-triggered-the-ope-alarm
tags: [ALIGNMENT, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: LessWrong post (via GreaterWrong mirror; TL;DR and sections 1-3.1 in full, later sections not read)
digest_buckets: []
importance: 1
---

# Every reward-hacked policy I tested triggered the OPE alarm — and so did my best honest one (JulesRoussel01 2026)

**What it does.** The testbed is Wordle with Qwen2.5-0.5B-Instruct over a closed list of 156 words, so action probabilities and importance ratios are exact. The author builds a faithful policy (SFT from a scripted teacher, then GRPO), a benign drift policy, and reward-hacked policies, which come from distilling a scripted proxy-farming teacher after GRPO fails to make them. It then asks whether importance-sampling off-policy evaluation (OPE) on logged trajectories can tell hacking from benign change.

**Key result.** Seven GRPO attempts on flawed but admissible proxies, one with no KL, produced no hacker: optimisation mostly made better Wordle players. Every certified hacker had effective sample size 0.00, but so did the best benign improver (0.03). Against a benign control matched on weight variance, 0 of 6 diagnostics were hacking-specific. One quantity, the share of actions at the logging policy's exploration floor, was ≤ 0.05 for benign policies and 0.15-0.16 for hackers. The author treats it as a candidate, not a detector.

**Quote.** "under a benign control matched on importance-weight variance, 0 of 6 diagnostics passed my pre-registered test for being hacking-specific." (TL;DR)

**Relevance to our paper.** Little bearing. It is a behavioural OPE study of a tiny policy, with no internals and no reuse question. Its one tangential echo is a methods lesson we share: a diagnostic that tracks "how far the policy moved" is not one that tracks "what kind of change", and it needs a control matched on distance (as our random-update floor is).

**For the digest.**
