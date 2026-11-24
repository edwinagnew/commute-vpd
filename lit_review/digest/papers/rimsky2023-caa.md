---
slug: rimsky2023-caa
title: "Steering Llama 2 via Contrastive Activation Addition"
authors: Rimsky et al.
year: 2023
id: 2312.06681
tags: [BASELINE, DIFFING, SUPPORTS]
primary_tag: BASELINE
in_our_bib: rimsky2024
cited_in_draft: yes
themes: [baselines, model-diffing]
read_from: "arXiv HTML: abstract, intro, method, sections 4-6 (results, system prompts, comparison to fine-tuning), section 8.3 (base vs chat vectors), discussion and limitations"
digest_buckets: [4-supports]
importance: 4
---

# Steering Llama 2 via Contrastive Activation Addition (Rimsky 2023)

**What it does.** Contrastive Activation Addition (CAA): the steering vector is the mean difference of residual
activations at the answer token over many A/B multiple-choice pairs that differ only in whether the answer shows a
behaviour (sycophancy, refusal, hallucination, myopic reward, corrigibility, survival instinct, AI coordination). The
vector is added at every position after the prompt with a signed multiplier. Main models Llama-2-7B/13B-Chat; vectors
are also built from Llama-2-7B base and compared with and transferred to the chat model; CAA is layered on top of
system prompts and on top of supervised fine-tuning.

**Key result.** Adding or subtracting the vector moves all seven behaviours in the expected direction on
multiple-choice and open-ended tests, with little change on MMLU. On top of fine-tuning it steers further for 3 of 7
behaviours, with some counter-intuitive interactions (positive refusal steering after fine-tuning lowers refusal).
Base and chat steering vectors have a cosine-similarity peak between layers 7 and 15, and base-model vectors steer the
chat model "significantly, especially between layers 10 and 15".

**Quote.** "This surprising trend indicates that RLHF has a smaller effect on how information is represented between
layers 7 and 15." (Section 8.3)

**Relevance to our paper.** Cited correctly as the mean-difference steering vector our steering share follows. The
base-to-chat transfer in section 8.3 is a small, activation-level piece of evidence that RLHF kept the base model's
behaviour directions in mid layers: support for EH read by value, not by mechanism, which our definition sets out to
replace. The layer dependence (agreement in mid layers, decay elsewhere) is a reminder that a steering share depends on
the layer chosen, which our method chooses per fine-tune on fitting windows. Not a known-answer case: no mechanism is
traced in either model.

**For the digest.**
- 4-supports: base-model behaviour vectors transfer to Llama-2-chat in layers 10-15, activation-level support for EH under RLHF.
