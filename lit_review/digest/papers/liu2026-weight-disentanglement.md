---
slug: liu2026-weight-disentanglement
title: Understanding and Enforcing Weight Disentanglement in Task Arithmetic
authors: Liu et al.
year: 2026
id: 2604.17078
tags: [BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [other]
read_from: "arXiv HTML: abstract, introduction, Sections 3-4.2 (definitions and theorems), Section 5 (results)"
digest_buckets: []
importance: 2
---

# Understanding and Enforcing Weight Disentanglement in Task Arithmetic (Liu 2026)

**What it does.** Explains when task vectors add without interference. Under the NTK linearisation, it defines
Task-Feature Specialization (TFS): each task's output is sensitive (by gradient) only to a disjoint set of a layer's
column "features". It proves TFS implies weight disentanglement and block orthogonality of weight columns, then
proposes OrthoReg, a regulariser that pushes each weight update ΔW towards orthogonal columns. Experiments: CLIP
ViT-B/32, B/16, L/14 on the eight-task image benchmark.

**Key result.** OrthoReg raises merged-model accuracy for every baseline, e.g. non-linear fine-tuning on ViT-L-14 from
84.07% to 88.23% absolute and normalised accuracy to 100.08%. It also improves task negation (an extra 10 points of
forgetting for ATT-FT on ViT-L-14 at fixed ImageNet accuracy) and makes task vectors more mutually orthogonal.

**Quote.** "we introduce Task-Feature Specialization (TFS), a model’s ability to allocate distinct internal features to different tasks, as the fundamental principle." (Abstract)

**Relevance to our paper.** Mostly background. Its "specialized feature set" (the columns the output is sensitive to on
a task's data) is a gradient-based cousin of our needed set K, which we estimate by interchange interventions
instead; this could be one sentence if we justify interventions over gradients. It does not ask whether a task
vector reuses base mechanisms.

**For the digest.**
- (no bucket)
