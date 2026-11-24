---
slug: kumar2022-finetuning-distort
title: Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution
authors: Kumar et al.
year: 2022
id: 2202.10054
tags: [SUPPORTS, FT-INTERP, DYNAMICS]
primary_tag: SUPPORTS
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, training-dynamics]
read_from: "arXiv HTML: abstract, introduction, Section 3 (theory, incl. 3.2 key intuitions), Section 4.3 (feature distortion checks)"
digest_buckets: [4-supports]
importance: 4
---

# Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution (Kumar 2022)

**What it does.** Asks why full fine-tuning beats linear probing in distribution but loses to it out of distribution.
Theory on a two-layer overparameterised linear network: gradient updates to the feature extractor B only move it
inside the subspace spanned by the training inputs, while the head is learned at the same time. Experiments on CLIP,
MoCo and other ResNet/ViT backbones over ten distribution-shift benchmarks, plus LP-FT (fit the head first, then
fine-tune everything).

**Key result.** Fine-tuning averages 2% higher ID accuracy but 7% lower OOD accuracy than linear probing; LP-FT gets
1% better ID and 10% better OOD than fine-tuning. Features of ID examples move more than OOD ones (mean feature change
0.0188 vs 0.0167 on Living-17), and LP-FT changes features about 20x less than fine-tuning (0.0011 ID). Early
stopping does not close the gap (LP 67.1% vs FT 61.3% OOD with OOD-chosen stopping).

**Quote.** "The only way the pretrained features are not distorted and only scaled during FT is if the initial feature extractor B0 is exactly aligned with the ID subspace." (Section 3.2.1)

**Relevance to our paper.** This is a clean theoretical reason to expect our finding that trained fine-tunes turn
writes rather than rescale them: in their model, a fine-tune is a pure rescale of the base features only in a
measure-zero case, and otherwise it distorts them inside the training-data subspace. It bears on our open question
of whether Def. 1's "kept up to a positive scalar" is too strict, and argues for the rotation-tolerant kept clause we
plan to test. LP-FT versus FT is also a ready-made pair of updates with similar ID behaviour but a 20x different
internal change, the kind of pair our claim "two fine-tunes with one input-output behaviour can differ" is about;
it could be reproduced cheaply on our 67M model (freeze-then-tune versus full tune on one task). It is vision and
linear-theory work, with no mechanism-level evidence of reuse, so it is not a known-answer case.

**For the digest.**
- 4-supports: theory predicts fine-tuning distorts (turns) base features rather than scaling them except in a degenerate case, matching our "turn not scale" reading; cite next to the kept-clause discussion.
