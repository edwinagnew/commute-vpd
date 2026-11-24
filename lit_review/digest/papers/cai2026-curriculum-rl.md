---
slug: cai2026-curriculum-rl
title: "Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model"
authors: Cai et al.
year: 2026
id: 2606.22317
tags: [EH, BACKGROUND]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "arXiv HTML full (abstract, sec. 3 method, sec. 4 results, appendix A)"
digest_buckets: []
importance: 3
---

# Curriculum Reinforcement Learning Can Incentivize Reasoning Capacity in LLMs Beyond the Base Model (Cai 2026)

**What it does.** Responds to Yue et al. 2025 (RLVR does not raise pass@k at large k). It argues that problems the model
never solves give zero-advantage GRPO groups and so no learning signal. Each of three rounds: find each problem's
first-success index under 256 samples, sort problems into five difficulty groups, run a small SFT "bridge" on 20 teacher
traces for problems near the pass@32 and pass@256 cutoffs, then GRPO on a chosen difficulty range. Models: Qwen2.5-7B
(base and Instruct), Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B, Llama-3.1-8B-Instruct; trained on DAPO-Math-17K, scored on
AIME 2024/2025 and MATH500.

**Key result.** Averaged over benchmarks, vanilla RLVR changes pass@256 by -0.5 points against the base model and the
curriculum by +9.8 points. Problems unsolved in 256 samples fall from 538 to 350 across the five settings, with 226
base-unsolved problems becoming solvable.

**Quote.** "Vanilla RLVR changes mean pass@256 by -0.5 percentage points on average, whereas Curriculum RL improves mean pass@256 by +9.8 percentage points." (sec. 4)

**Relevance to our paper.** Behavioural evidence on the elicitation-versus-expansion question at the RL end. It confirms
Yue et al.'s finding for vanilla RLVR. The expansion it reports comes with teacher SFT, so it says that external traces
can move the pass@k boundary, not that RL alone builds new mechanisms. It is pass@k only, with no look inside the model,
which is the gap our draft names. At most a citation in the RL-scaling paragraph the draft asks for; no method or
known-answer case.

**For the digest.** (none)
