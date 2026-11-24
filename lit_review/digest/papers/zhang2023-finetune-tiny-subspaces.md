---
slug: zhang2023-finetune-tiny-subspaces
title: "Fine-tuning Happens in Tiny Subspaces: Exploring Intrinsic Task-specific Subspaces of Pre-trained Language Models"
authors: Zhang et al.
year: 2023
id: 2305.17446
tags: [DYNAMICS, BACKGROUND]
primary_tag: DYNAMICS
in_our_bib: no
cited_in_draft: no
themes: [training-dynamics, interpreting-finetunes]
read_from: "arXiv HTML full (abstract, method sec. 3, results sec. 4.1-4.5)"
digest_buckets: []
importance: 3
---

# Fine-tuning Happens in Tiny Subspaces: Exploring Intrinsic Task-specific Subspaces of Pre-trained Language Models (Zhang 2023)

**What it does.** Asks whether fine-tuning of BERT-base and RoBERTa-base on GLUE lies in a low-dimensional,
task-specific subspace. It saves one checkpoint per epoch of an ordinary fine-tune, runs an SVD on the stacked
trajectory per encoder layer, and then re-trains the model with only the coefficients in that 32-dimensional (64 for
MNLI) subspace trainable. It also builds an 8-dimensional "unified" subspace from one checkpoint per task, and zeroes the
largest entries ("outlier dimensions") of the learned update.

**Key result.** Fine-tuning in the trajectory subspace nearly matches full fine-tuning (GLUE average 81.21 vs 82.13 for
BERT, 84.40 vs 85.43 for RoBERTa), while a random subspace of the same size barely beats a frozen encoder (72.54).
Subspaces transfer partly between tasks. The update has a few outlier weights, mostly in LayerNorm gains and dense
output weights; zeroing them collapses performance (BERT CoLA 59.37 to 0.00, SST-2 91.95 to 50.92), while zeroing as
many random weights does almost nothing.

**Quote.** "Fine-tuning happens only in some tiny task-specific subspaces, which can be effectively learned with a small number of free parameters." (sec. 1)

**Relevance to our paper.** Background for our effective-rank reading and for the observation that fine-tune updates are
low-dimensional. Its subspace is found from the fine-tune's own trajectory, not from the base model, so it does not say
whether that subspace is a base mechanism (reuse) or a new one. The finding that a handful of LayerNorm-gain weights in
the update carry the task is mildly suggestive for our gain-only (IA3-style) refits, but this is an encoder
classification setting with a fresh head, far from ours. No known-answer case, no definition.

**For the digest.** (none)
