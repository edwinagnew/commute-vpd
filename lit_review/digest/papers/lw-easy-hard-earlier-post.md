---
slug: lw-easy-hard-earlier-post
superseded_by: easy-hard-2602-07852 (earlier LessWrong post of the same work; cite the paper)
title: "Narrow Misalignment is Hard, Emergent Misalignment is Easy"
authors: Edward Turner et al.
year: 2025
id: https://www.lesswrong.com/posts/gLDSqQm8pwNiq7qst
tags: [FT-INTERP, EH, ALIGNMENT, SUPPORTS]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, elicitation]
read_from: LessWrong post full (via GreaterWrong mirror)
digest_buckets: [4-supports]
importance: 2
---

# Narrow Misalignment is Hard, Emergent Misalignment is Easy (Turner 2025)

**What it does.** This is the July 2025 research update (Turner, Soligo, Rajamanoharan, Nanda) that the arXiv paper 2602.07852 later expanded; see that entry. It trains bad-medical-advice steering vectors and rank-1 and rank-32 LoRAs at layer 24 of a 48-layer Qwen model. A KL penalty on other-domain data gives a narrow solution, which is compared with the general one on efficiency (loss vs norm) and stability (loss under orthogonal noise).

**Key result.** Without KL, the fine-tune gives about 40% misaligned general answers and about 60% bad medical advice. With KL it gives bad medical advice up to 65% of the time and 0% general misalignment. Mixing bad medical advice with good advice in other domains still gave EM down to 1/6 bad data. The general solution is more efficient and more stable, and continuing training without KL from the narrow solution reverts to the general one. The post also introduces a "Training Lens" (PCA of steering-vector checkpoints). Its top 3 PCs (79% of variance) are not enough to reproduce the behaviour. A narrow solution can also come from multi-epoch training on about 4 (rank 1) or 15-25 (rank 32) unique samples.

**Quote.** "The general misalignment solution is consistently more stable and more efficient than the narrow solution." (TL;DR)

**Relevance to our paper.** It is superseded by 2602.07852 and should be cited as that paper. The pre-training-significance result that bears most on the elicitation hypothesis appears only in the paper version.

**For the digest.**
- 4-supports: early version of the "fine-tunes prefer the base-native general direction" finding; cite the arXiv paper instead.
