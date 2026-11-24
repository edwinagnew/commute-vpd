---
slug: wu2025-shadow-ft
title: "Shadow-FT: Tuning Instruct Models via Training on Paired Base Model"
authors: Wu et al.
year: 2025
id: 2505.12716
tags: [BACKGROUND, EH]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [elicitation, other]
read_from: arXiv HTML abstract+intro+section 2 (weight similarity)+method+main results+sections 5.1, 5.3, 5.6
digest_buckets: []
importance: 2
---

# Shadow-FT: Tuning Instruct Models via Training on Paired Base Model (Wu 2025)

**What it does.** Observes that further fine-tuning an instruction-tuned LLM often helps little or hurts, while the base
and instruct weights differ little (relative gap σ = Σ|W_B − W_I| / (Σ|W_B| + Σ|W_I|), 0.016 on average for
Llama-3.1-8B). Shadow-FT fine-tunes the base model instead (full or LoRA, also DPO) and adds the learned weight delta to
the instruct model. Tested on Qwen 2.5/3 and Llama 3 models (1B to 32B) over 19 maths, code and knowledge benchmarks.

**Key result.** Shadow-FT beats ordinary fine-tuning of the instruct model across sizes; e.g. on Qwen-3-4B an average of
69.4 against 65.7 for conventional tuning and 68.0 for the untuned instruct model; with LoRA, 4.7/10.1/3.9 points above
conventional LoRA on Math-7/Code-3/Knowledge-9. At the start of tuning Qwen3-8B, the instruct model's loss is 22.6%
higher and its gradient 3.25x larger than the base model's.

**Quote.** "The Base model tends to be a good learner yet a weak backbone without post-training." (Abstract)

**Relevance to our paper.** Mostly background. Two loose points of contact: (i) it is one more measurement that
instruct and base weights are close (σ ≈ 0.016), consistent with the small-update premise behind treating
post-training as a modest change to the base; (ii) that a delta learned on the base transfers to the instruct model is
behavioural evidence that the two share enough structure for updates to commute, which is adjacent to our project's
original question of carrying an update across. It offers no mechanism-level evidence and no reading of what a
fine-tune changed, so it is not a baseline or known-answer case. Cite only if the draft wants a weight-closeness number.

**For the digest.** (none)
