---
slug: ryu2023-lowrank-residual-storage
title: "Efficient Storage of Fine-Tuned Models via Low-Rank Approximation of Weight Residuals"
authors: Ryu et al.
year: 2023
id: 2305.18425
tags: [FT-INTERP, DYNAMICS, BACKGROUND]
primary_tag: BACKGROUND
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, other]
read_from: arXiv HTML abstract, sections 1-5 (analysis of residuals, method, GLUE / image / Dolly results)
digest_buckets: []
importance: 2
---

# Efficient Storage of Fine-Tuned Models via Low-Rank Approximation of Weight Residuals (Ryu 2023)

**What it does.** A compression paper. It studies the weight residual Δθ = θ' − θ of full fine-tunes (RoBERTa-Large on
GLUE, Stable Diffusion and StyleGAN style fine-tunes, Dolly-3B/7B instruction fine-tunes of Pythia) and stores it as
a truncated SVD with per-layer rank budgets and 4-bit quantisation of the singular vectors (ERE).

**Key result.** The residual's effective rank falls fast during fine-tuning and is much lower than the base weights';
truncating the residual changes outputs far less than truncating the base weights. A rank-4 residual (13 MB) keeps
Dolly-7B's instruction following and scores slightly above both Dolly and Pythia on zero-shot benchmarks (average 59.8
vs 58.9 and 56.9). RoBERTa on GLUE keeps 89.2 vs 89.4 average at 6% of the storage.

**Quote.** "weight residuals of overparameterized models, already acknowledged for their low-rank nature, exhibit even more pronounced low-rank characteristics." (§1)

**Relevance to our paper.** Bears lightly on our effective-rank reading of an update: it is an early systematic
measurement that full fine-tune updates are low-rank, and that instruction following in Dolly survives at rank 4. The
Dolly result is weak behavioural support for elicitation (a tiny update unlocks instruction following), but the
paper gives no mechanism and the authors' own reading is informal. Not a method we need to compare against. At most a
citation beside other low-rank-update evidence if the draft discusses effective rank.

**For the digest.** (none)
