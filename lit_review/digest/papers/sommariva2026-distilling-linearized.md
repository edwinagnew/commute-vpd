---
slug: sommariva2026-distilling-linearized
title: Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic
authors: Sommariva et al.
year: 2026
id: 2605.18993
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: arXiv HTML abstract+intro+background+method+experiments+model analysis+conclusions (math stripped, so table numbers not read)
digest_buckets: []
importance: 1
---

# Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic (Sommariva 2026)

**What it does.** A model-merging paper. Fine-tuning in the tangent space of a pre-trained model (linear fine-tuning,
Ortiz-Jimenez et al. 2023) gives task vectors that add and subtract with little interference, but it is costly and
less expressive. DELTA trains an ordinary non-linear student by distilling hidden activations from a linearized teacher
(sampled along the path from the pre-trained to the current weights), plus a KFAC curvature penalty on representation
drift estimated on a reference dataset. Backbones are CLIP ViT-B/32 and ViT-L/14 (8- and 14-task vision benchmarks)
and T5-base (6 NLI tasks).

**Key result.** DELTA beats the in-training merging baselines on absolute accuracy for task addition, is robust to
the task-vector scaling coefficient, and is competitive on negation (beaten by methods that stay in the linear
regime). The student has near-zero "linearization error" under weight perturbation and a stronger in-domain vs
out-of-domain separation of activation change; distillation drives the first, the curvature penalty the second.
Training costs ~3x time and 2x memory.

**Quote.** "task arithmetic does not require strict linearization, but rather localized and approximately linear
update directions." (Introduction)

**Relevance to our paper.** Little direct bearing. It is about engineering fine-tunes to be composable, not about
whether a fine-tune reuses a base mechanism; its "support localization" is an input-space notion (the update changes
activations only on its own task's inputs), not a mechanism-level one. At most it is context for the idea that a
fine-tune can be pushed to stay near the base model's tangent space, which is loosely related to our "kept up to a
gain" clause, but the paper makes no claim about mechanisms.

**For the digest.** (none)
