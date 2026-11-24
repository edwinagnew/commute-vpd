---
slug: rinaldi2026-gradfix
title: Gradient-Sign Masking for Task Vector Transport Across Pre-trained Models (GradFix)
authors: Rinaldi et al.
year: 2026
id: https://proceedings.iclr.cc/paper_files/paper/2026/hash/59e4f5d009f95324a1a1aa9971b51e25-Abstract-Conference.html
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "ICLR proceedings PDF pages 1-9 (abstract, method, Tables 1-2)"
digest_buckets: []
importance: 1
---

# Gradient-Sign Masking for Task Vector Transport Across Pre-trained Models (Rinaldi 2026)

**What it does.** Moves a task vector fine-tuned on one pre-trained model onto a different pre-training of the same
architecture (CLIP DataComp-XL to LAION-2B; T5 v1.1 to FLAN-T5). The source task vector is masked elementwise,
keeping only coordinates whose sign agrees with the target model's anti-gradient estimated by majority vote over a
few labelled examples; a first-order argument shows the masked update is a descent direction.

**Key result.** Naive addition of the source vector stays at zero-shot level (ViT-B/16 EuroSAT 49.6 vs zero-shot
49.4). With one example per class GradFix reaches 61.9 on EuroSAT and 71.1 on SVHN, beating few-shot fine-tuning in
most cells; with an oracle mask from the true target task vector it reaches 95.1 on EuroSAT. T5 average rises from
42.1 (naive) to 58.5, against 89.4 for full fine-tuning.

**Quote.** "In this work, we show that successful transfer depends strongly on the gradient-sign structure of the new model." (Abstract)

**Relevance to our paper.** Little direct bearing: it transports updates across different pre-trainings in raw
coordinates, with no decomposition and no question of reuse inside one base model. The earlier summary's account of
the method and numbers matches the paper.

**For the digest.**
- (no bucket)
