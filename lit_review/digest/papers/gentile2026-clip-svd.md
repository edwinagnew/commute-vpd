---
slug: gentile2026-clip-svd
title: "From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition"
authors: Gentile et al.
year: 2026
id: 2603.24653
tags: [FT-INTERP, DIFFING, DECOMP, EH, CONTRADICTS]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, model-diffing, decompositions]
read_from: arXiv HTML abstract, method (3.1-3.3), section 6 (model adaptation), conclusion; sections 4-5 skimmed
digest_buckets: [4-contradicts, 1-missing-method]
importance: 4
---

# From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition (Gentile 2026)

**What it does.** SITH takes each CLIP ViT attention head's value-output matrix, factors it by SVD, and labels each
singular vector with a sparse, coherent set of text concepts (COMP, a matching-pursuit variant in CLIP's joint space).
It uses the labels to edit singular values and remove spurious or NSFW concepts. In Sec. 6 it fine-tunes CLIP
ViT-L/14 on Flowers 102, Oxford Pets and CUB-200, by full fine-tuning and by LoRA, and compares the heads' weights.

**Key result.** The right-singular subspaces of the VO matrices before and after fine-tuning have high normalised
spectral cosine similarity (Fig. 7). Most singular vectors of the difference ΔW are labelled with concepts from the
fine-tuning domain, judged by an LLM (Fig. 8), for example plant terms for Flowers 102. The paper gives these figures,
not a share of the update that is a rescaling.

**Quote.** "we use SITH to study model adaptation, showing how fine-tuning primarily reweights a stable semantic basis
rather than learning entirely new features." (Abstract)

**Relevance to our paper.** The paper makes the claim our trained-fine-tune result argues against, that fine-tuning
reweights a fixed basis, but its evidence is weaker than its wording. Stable singular subspaces are expected of any
small update. Their own task singular vectors are new, task-specific directions, which is closer to our "turns writes"
finding than to pure reweighting. To contradict us it would need the share of ΔW on the diagonal of the base SVD basis,
which is our rescale share at the head level. That reading is cheap on our 67M heads (data-free SVD of W_VO before and
after; COMP labels need CLIP's text space and do not transfer). Cite it as a face-value counter-claim, and state what
its measurement can and cannot tell apart.

**For the digest.**
- 4-contradicts: claims CLIP fine-tunes "reweight a stable semantic basis"; evidence is subspace cosine, which does not separate rescaling from small rotations.
- 1-missing-method: per-head SVD subspace stability plus a task-SVD reading of ΔW is a weight-only diff we could run on the 67M as a cheap reference.
