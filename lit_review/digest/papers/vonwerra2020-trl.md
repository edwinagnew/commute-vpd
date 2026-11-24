---
slug: vonwerra2020-trl
title: "TRL: Transformers Reinforcement Learning"
authors: von Werra et al.
year: 2020
id: https://github.com/huggingface/trl
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: vonwerra2020trl
cited_in_draft: yes (only in appC_rl_sft_qwen.tex, which main.tex does not input)
themes: [other]
read_from: "GitHub README (main) and trl/trainer/grpo_config.py on main, fetched 2026-11-23"
digest_buckets: []
importance: 1
---

# TRL: Transformers Reinforcement Learning (von Werra 2020)

**What it does.** A Hugging Face library for post-training transformer models: trainers for SFT, GRPO (sync and
async), DPO, KTO and others, built on Transformers with Accelerate/DeepSpeed scaling and PEFT (LoRA/QLoRA)
integration. Software, not a paper.

**Key result.** No results. Relevant defaults in `GRPOConfig` on main: `beta = 0.0` ("If `0.0` (default), the
reference model is not loaded"), `loss_type = "dapo"` (normalise by active tokens in the global batch; `"dr_grpo"`
normalises by a global constant), `scale_rewards = "group"` (divide by the group's std; `"none"` follows Dr. GRPO).

**Quote.** "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving
training speed." (GRPOConfig docstring)

**Relevance to our paper.** Cited only for "no KL penalty to the base model, as in TRL's default", which matches the
current code. Note that TRL's defaults are otherwise not Dr. GRPO (std scaling on, DAPO token normalisation), so the
sentence should keep attributing only the KL choice to TRL. Defaults in a library change between releases (only main was
read here), so the sentence should name the TRL version used. Like the RL recipe citation, it sits in an appendix main.tex does not
input. No bearing on our claims.

**For the digest.**
