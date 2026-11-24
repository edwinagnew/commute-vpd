---
slug: liu2025-prorl
title: "ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models"
authors: Liu et al.
year: 2025
id: 2505.24864
tags: [EH, CONTRADICTS, KNOWN-ANSWER, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, training-dynamics]
read_from: "local PDF arxiv_2505.24864.pdf (v1) via pdf-text: abstract, §1, §2 (GRPO, KL control, reference reset), §4.1-4.3 (boundary analysis, creativity index, diminish/plateau/sustained); appendices not read"
digest_buckets: [2-known-answer-case, 4-contradicts, 5-cite-fix]
importance: 4
---

# ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models (Liu 2025)

**What it does.** Trains DeepSeek-R1-Distill-Qwen-1.5B for more than 2k GRPO steps (with DAPO tricks, a KL penalty and
periodic resets of the reference policy) on maths, code, STEM, logic puzzles and instruction following. It releases
the result as Nemotron-Research-Reasoning-Qwen-1.5B. It compares pass@k curves of base and RL model per task, and
measures novelty with the Creativity Index (overlap of outputs with the DOLMA pre-training corpus).

**Key result.** Average pass@1 gains over the base are 14.7% on maths, 13.9% on code, 54.8% on logic puzzles, 25.1% on
STEM and 18.1% on instruction following. On some tasks the base model never succeeds and the RL model reaches 100%. The
gain in pass@128 is largest where the base model's pass@128 is low, and negative or flat where it is high (many maths
benchmarks: "diminished" boundary). Tasks with little gain have base outputs with low creativity index (heavily seen in
pre-training). Creativity index rises with prolonged training (mean 3.84 for the base, 4.70 for the final model).

**Quote.** "RL expands a model’s reasoning boundary most effectively in domains where the base model initially
struggles." (§1)

**Relevance to our paper.** This is the standard counterpoint to Yue et al. (2025), which our draft cites for the
selection view. The draft asks for more citations on RL scaling, and ProRL is the natural one: it should be cited
alongside Yue. Its per-task split (diminish / plateau / sustained) on one released base/RL pair offers a within-model
contrast: tasks where RL only sharpened, and tasks where pass@128 went from 0 to high. That is a behavioural
known-answer candidate on a real 1.5B model. The base is a Qwen2.5-architecture distilled model, not our Qwen3-1.7B,
so our decomposition would not transfer. Evidence is behavioural only. The creativity index is an output-novelty
measure, not a mechanism one.

**For the digest.**
- 2-known-answer-case: one released base/RL pair (R1-Distill-Qwen-1.5B → Nemotron-Research-Reasoning-Qwen-1.5B) with tasks that are behaviourally "sharpened" and tasks that are "new" (pass@128 0 → 100%). It needs a decomposition of that base.
- 4-contradicts: behavioural evidence that long RL extends the reasoning boundary where base coverage is low, against "post-training only re-weights".
