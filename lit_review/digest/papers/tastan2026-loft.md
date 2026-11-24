---
slug: tastan2026-loft
title: "LOFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning"
authors: Tastan et al.
year: 2026
id: 2505.21289
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: arXiv HTML (2505.21289; same paper as the ICLR 2026 entry) abstract, method 2.1-2.4, results 3.1-3.2, conclusion
digest_buckets: []
importance: 1
---

# LOFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning (Tastan 2026)

**What it does.** A LoRA optimizer that tries to follow the full-fine-tuning AdamW path. It updates one factor at a
time, scales gradients by the inverse Gram matrix of the other factor, recalibrates the first and second Adam moments
when the subspace changes, and clips the projected full gradient. Tested on LLaMA-7B/2-7B/3-8B commonsense reasoning
and ViT-Base image classification.

**Key result.** On LLaMA-7B at rank 16, average accuracy is 76.08% against 73.57% for LoRA and 71.11% for DoRA. At
rank 1 it is 72.17%. On ViT-Base, rank 16 gives 76.12% against 75.86% for full fine-tuning. The authors call it the
first low-rank method that reduces exactly to AdamW at full rank.

**Quote.** "LoFT is the first low-rank adaptation method that exactly reduces to AdamW ... in the full-rank limit."
(Sec. 1, contributions)

**Relevance to our paper.** Little direct bearing. It is an optimizer for low-rank fine-tuning and says nothing about
mechanisms, reuse or diffing. The projection of a full gradient onto a factor's subspace is related in form to carrying
an update onto subcomponents by least squares, but our split does not need it.

**For the digest.** (no bucket)
