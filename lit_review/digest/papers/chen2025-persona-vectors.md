---
slug: chen2025-persona-vectors
title: "Persona Vectors: Monitoring and Controlling Character Traits in Language Models"
authors: Chen et al.
year: 2025
id: 2507.21509
tags: [BASELINE, DIFFING, EH, ALIGNMENT, CITE-FIX]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, interpreting-finetunes, elicitation]
read_from: "arXiv HTML v3: abstract, sections 2-6 (extraction, steering, finetuning shift, preventative steering, data screening), appendix C.2 table"
digest_buckets: [1-missing-method, 4-supports]
importance: 5
---

# Persona Vectors: Monitoring and Controlling Character Traits in Language Models (Chen 2025)

**What it does.** An automated pipeline turns a natural-language trait description (evil, sycophancy, hallucination, and others) into contrastive system prompts. A "persona vector" is the difference in mean residual activation between responses that show the trait and responses that do not. On Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, the paper fine-tunes on trait-eliciting and EM-like datasets (insecure code, flawed maths, and so on). It then measures the "finetuning shift": the mean last-prompt-token activation of the fine-tuned model minus that of the original, projected onto the persona vector extracted from the original model.

**Key result.** The finetuning shift along a persona vector correlates with post-fine-tune trait expression at r = 0.76 to 0.97, against cross-trait baselines of r = 0.34 to 0.86. Subtracting the vector at inference reduces the trait but costs MMLU accuracy. "Preventative steering", which adds the vector during training, limits the trait shift and better preserves capability. Before training, a dataset-level "projection difference" predicts the shift, and it also flags problematic samples.

**Quote.** "We observe strong positive correlations (r = 0.76–0.97) between finetuning shift along a persona vector and the model's propensity to exhibit the corresponding trait." (Sec. 4.2)

**Relevance to our paper.** The finetuning shift is a cheap reading of an update. It projects the activation difference between the base and fine-tuned models onto a direction fixed from the base model. That is close to our mean-difference steering share and to Minder et al.'s activation-difference baseline, and it could run on our 67M fine-tunes wherever a contrast set defines the target behaviour. On EH, it is evidence at the activation level that fine-tune-induced traits move along directions already present in the pre-fine-tune model. Preventative steering is a weak causal check: supplying the direction removes the pressure to learn it. The evidence reads activations by value, not mechanisms, so it is exactly the kind our gap sentence says "reads activations". Our finding that trained fine-tunes turn writes rather than rescale them is not contradicted, because a shift along an existing direction can come from rotated weights. Our related work on model diffing should cite it next to Wang et al. 2025.

**For the digest.**
- 1-missing-method: the finetuning shift (activation shift projected onto a base-model trait direction) is a cheap diffing reading that we do not run; it is near our steering share and could be scored on our known answers.
- 4-supports: fine-tune-induced traits move along directions already in the pre-fine-tune model (r up to 0.97); this supports the elicitation hypothesis at the activation level, with no evidence at the level of mechanisms.
