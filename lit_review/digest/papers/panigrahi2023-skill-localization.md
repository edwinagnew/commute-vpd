---
slug: panigrahi2023-skill-localization
title: "Task-Specific Skill Localization in Fine-tuned Language Models"
authors: Abhishek Panigrahi et al. (Saunshi, Zhao, Arora)
year: 2023
id: 2302.06600
tags: [DIFFING, BASELINE, DECOMP, FT-INTERP]
primary_tag: BASELINE
in_our_bib: panigrahi2023
cited_in_draft: yes
themes: [model-diffing, baselines, interpreting-finetunes]
read_from: "arXiv HTML: abstract, Sec. 1, 2 (grafting definition and optimisation), 3.1-3.2 (results, AdamW), 4.3, 7"
digest_buckets: [1-missing-method]
importance: 4
---

# Task-Specific Skill Localization in Fine-tuned Language Models (Panigrahi 2023)

**What it does.** This ICML 2023 paper defines *model grafting*: a binary mask γ copies the fine-tuned values of a few
parameters onto the pretrained model, θ_graft = θ_pre + γ ⊙ (θ_ft − θ_pre). The mask is learned by a short optimisation,
starting from the most-moved parameters, so that the grafted model recovers the fine-tune's performance. The models are
RoBERTa-base with prompt-based fine-tuning (SGD) on 13 mostly GLUE tasks, plus GPT-2. Multi-task and continual-learning
settings are also tested.

**Key result.** Grafting about 0.01% of parameters (fewer than 8,500) recovers more than 95% of fine-tuned accuracy, and
grafted models agree with the fine-tune on 93% (single-sentence) and 86% (two-sentence) of test labels. Grafted models are
better calibrated (40-90% lower ECE) and often better out of distribution. Grafting the top-moved parameters does much
worse than the learned mask at low sparsity. Fine-tuning with AdamW does not yield sparse grafts unless an ℓ1 penalty on
parameter movement is added. In multi-task training the regions for different tasks are almost disjoint.

**Quote.** "a very small subset of parameters (∼0.01% of model parameters) responsible for (>95%) of the model's
performance" (abstract)

**Relevance to our paper.** The draft cites it accurately as a weight-level method that finds "where an update sits"
without asking whether what it computes is old. The appendix also names grafting as the closest precedent for a reading
that swaps in fine-tuned pieces. It could be run on our 67M fine-tunes as a baseline: learn a sparse graft and read graft
size or location against the built pairs. Two cautions. Sparse grafts fail for AdamW fine-tunes without an ℓ1 penalty,
and ours are presumably Adam, so localisation may simply fail. And a small graft region says the update is compact, not
that it reuses a base mechanism, so it is a natural update-size-type baseline, not a reuse reading. It gives no evidence
on reuse versus new.

**For the digest.**
- `1-missing-method`: sparse grafting is runnable on our fine-tunes as a weight-level baseline, but it measures compactness, not reuse, and fails for AdamW without ℓ1.
