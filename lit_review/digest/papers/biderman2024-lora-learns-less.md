---
slug: biderman2024-lora-learns-less
title: "LoRA Learns Less and Forgets Less"
authors: Biderman et al.
year: 2024
id: 2405.09673
tags: [FT-INTERP, BASELINE, SUPPORTS]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, baselines]
read_from: "arXiv HTML (TMLR version): abstract, intro, setup, section 4 results incl. 4.6 spectral analysis and its limitations, conclusion"
digest_buckets: [4-supports]
importance: 4
---

# LoRA Learns Less and Forgets Less (Biderman 2024)

**What it does.** Compares LoRA (ranks 16, 64, 256, all modules, α = 2r) with full fine-tuning of Llama-2-7B on code
and maths, in continued pretraining (StarCoder-Python, OpenWebMath, up to 20B tokens) and instruction tuning
(Magicoder-Evol-Instruct-110K, MetaMathQA). Learning is HumanEval / GSM8K; forgetting is HellaSwag, WinoGrande,
ARC-Challenge. Section 4.6 takes the SVD of base, fine-tuned and difference matrices across checkpoints.

**Key result.** LoRA underperforms full fine-tuning at common ranks, especially in continued pretraining (code CPT:
best LoRA 0.224 HumanEval vs 0.263 full); high-rank LoRA can match it in instruction tuning. LoRA forgets less, more
than dropout or weight decay achieve. Full fine-tuning barely changes the base spectrum, yet its difference matrix is
high rank: the rank for 90% of the variance is 10-100× typical LoRA ranks from the first 0.25B tokens, grows with
training, and is higher in MLPs than attention. A similar spectrum arises from adding small Gaussian noise.

**Quote.** "full finetuning learns perturbations with a rank that is 10-100× greater than typical LoRA
configurations" (Abstract)

**Relevance to our paper.** Bears on two of our readings. Our effective-rank reading assumes rank says something about
the update; their result that a trained update's spectrum looks like that of small Gaussian noise is the same warning
our random-update floor gives (ρ "tells an update from noise, not reuse from new"), and their own caveat that "the rank
needed to reconstruct the weight matrix is higher than the rank needed for a downstream task" applies to our rank
reading. A high-rank, noise-like full fine-tune is consistent with our finding that trained fine-tunes turn writes
rather than rescale them, though the paper does not test rescaling. It is also the reference for our Adam vs SGD vs
rank-8 LoRA comparison on the emoticon task (LoRA learns less, keeps the base model closer). No mechanism is traced,
so it is not a known-answer case.

**For the digest.**
- 4-supports: full fine-tune updates are high rank with noise-like spectra; consistent with our trained updates reading as noise-like and turning rather than rescaling, and a caution on reading effective rank.
