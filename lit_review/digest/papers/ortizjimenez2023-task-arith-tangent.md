---
slug: ortizjimenez2023-task-arith-tangent
title: "Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models"
authors: Ortiz-Jimenez et al.
year: 2023
id: 2305.12827
tags: [BACKGROUND, DIFFING, DEFINITION]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other, training-dynamics]
read_from: arXiv HTML abstract, sections 1, 3-6 and 8; appendices not read
digest_buckets: []
importance: 2
---

# Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models (Ortiz-Jimenez 2023)

**What it does.** Asks why adding and negating task vectors (θ* − θ₀) works in CLIP ViTs (B/32, B/16, L/14) fine-tuned
on eight image tasks. It tests whether fine-tuning stays in the linear (NTK) regime by post-hoc linearization. It
defines weight disentanglement: distinct task vectors change the function only on their own input regions. It also
fine-tunes directly in the tangent space and relates the results to localization of NTK eigenfunctions.

**Key result.** CLIP does not fine-tune in the linear regime: post-hoc linearized models lose single-task accuracy.
They are, however, more weight-disentangled. Fine-tuning in the tangent space keeps the disentanglement and improves
task addition by up to 5.8 points and task negation by up to 13.1 points. Task arithmetic fails from a random
initialisation, so the authors conclude that weight disentanglement is acquired in pre-training.

**Quote.** "the sole requirement for task arithmetic is actually weight disentanglement, where distinct directions in
weight space correspond to changes of the network in disjoint regions of the input space" (Sec. 1)

**Relevance to our paper.** Little direct bearing. The finding that real fine-tunes are not first-order in the weights
is a caveat on any reading of an update through one linearization at M. Our least-squares split is a weight-space fit,
not a Taylor expansion, so this does not apply to it directly. "Weight disentanglement emerges in pre-training" is
loosely in the spirit of the elicitation hypothesis, but it concerns how updates compose, not whether they reuse
mechanisms. It is optional background for the task-vector side of model diffing.

**For the digest.** (no bucket)
