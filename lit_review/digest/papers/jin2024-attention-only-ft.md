---
slug: jin2024-attention-only-ft
title: "Fine-Tuning Attention Modules Only: Enhancing Weight Disentanglement in Task Arithmetic"
authors: Jin et al.
year: 2024
id: 2407.07089
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: arXiv HTML abstract+intro+preliminaries+section 3 (kernel behaviour test, accuracy)+section 4.1 (disentanglement error)+discussion
digest_buckets: []
importance: 1
---

# Fine-Tuning Attention Modules Only: Enhancing Weight Disentanglement in Task Arithmetic (Jin 2024)

**What it does.** A model-merging paper on CLIP ViTs (B/32, B/16, L/14) fine-tuned on the standard 8 image
classification tasks of Ilharco et al. It proposes fine-tuning only the attention modules, as a cheaper alternative to
NTK-linearized fine-tuning, and tests "kernel behaviour" by comparing each fine-tuned model's accuracy with that of its
post-hoc first-order Taylor linearization. It then separates the representation module from the classification head
and measures a "disentanglement error" for pairs of task vectors under scaling.

**Key result.** Attention-only fine-tunes sit closer to their linearization than full fine-tunes (kernel behaviour),
keep individual-task accuracy near full fine-tuning, and merge better than both non-linear and NTK-linearized
fine-tuning. Weight disentanglement holds over a wide range of scaling coefficients in the representation module and
is degraded mainly by the task-specific classification heads.

**Quote.** "weight disentanglement mostly comes from the representation module, while the effectiveness of task
arithmetic is constrained by task-specific components like classification heads." (Introduction, contributions)

**Relevance to our paper.** Little direct bearing: it is about making task vectors composable in vision models, and
says nothing about whether a fine-tune reuses an internal mechanism. The "kernel behaviour" test (does the fine-tune
act like its linearization around the base) is a property of the update's size, not of mechanisms, so it is not a
reuse reading we need to compare against.

**For the digest.** (none)
