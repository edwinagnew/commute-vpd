---
slug: hu2021-lora
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: Hu et al.
year: 2021
id: 2106.09685
tags: [BASELINE, FT-INTERP, SUPPORTS, CITE-FIX]
primary_tag: CITE-FIX
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, baselines]
read_from: "arXiv HTML (v2): abstract, method (section 4), section 7 (which matrices, optimal rank, how ΔW compares to W) incl. Table 7"
digest_buckets: [1-missing-method, 4-supports, 5-cite-fix]
importance: 5
---

# LoRA: Low-Rank Adaptation of Large Language Models (Hu 2021)

**What it does.** Freezes pretrained weights and trains a low-rank update ΔW = BA (B initialised at zero) on chosen
matrices, mostly W_q and W_v; evaluated on RoBERTa, DeBERTa, GPT-2 and GPT-3 175B. Section 7 studies the learned
updates on GPT-3: which matrices to adapt, how small r can be, and how ΔW relates to W, by projecting W onto ΔW's top-r
singular subspace (‖UᵀWVᵀ‖_F) and comparing with W's own top-r directions and a random subspace.

**Key result.** LoRA matches or beats full fine-tuning with up to 10,000× fewer trainable parameters; very low ranks
(r = 1 or 2 on GPT-3 for WikiSQL and MultiNLI) suffice. At layer 48 of GPT-3 with r = 4, W projected on ΔW_q's subspace has norm 0.32, against 21.67
on W_q's top directions and 0.02 on a random subspace, while ‖ΔW_q‖_F = 6.91: an "amplification factor" of about 21.5.

**Quote.** "ΔW amplifies some features that are already in W ... instead of repeating the top singular directions of
W, ΔW only amplifies directions that are not emphasized in W" (Section 7.3)

**Relevance to our paper.** Our draft trains a rank-8 LoRA on the emoticon task (main.tex, figure on Adam vs SGD vs
LoRA) but does not cite LoRA; it should. Section 7.3 is also an early, weight-level reading of reuse: an update counts
as building on the base model when its subspace overlaps W's more than chance, and the overlap is with directions W
holds weakly. That reading is a $0 baseline on any of our updates and the obvious precursor to our least-squares
carrying of an update onto VPD subcomponents; it would say "reuse" for an update that amplifies weak existing
directions, which is what our "kept up to a positive scalar" clause makes precise per variable. Its finding fits
EH in weight space, but it does not separate rescaling an existing direction from rotating into a weakly present one,
the distinction our turning-not-scaling result depends on.

**For the digest.**
- 1-missing-method: the ΔW-subspace projection of W (vs top-r and random) is a cheap weight-space reuse reading we could report on all updates.
- 4-supports: LoRA updates amplify directions already present but weak in W, weight-level support for EH (not a rescale test).
