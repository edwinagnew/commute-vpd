---
slug: bussmann2024-batchtopk
title: "BatchTopK Sparse Autoencoders"
authors: Bussmann et al.
year: 2024
id: 2412.06410
tags: [BACKGROUND, BASELINE]
primary_tag: BACKGROUND
in_our_bib: bussmann2024batchtopk
cited_in_draft: yes
themes: [decompositions, baselines]
read_from: "arXiv HTML full text (short workshop paper: method, results, appendix A)"
digest_buckets: []
importance: 3
---

# BatchTopK Sparse Autoencoders (Bussmann 2024)

**What it does.** Replaces the per-sample top-k of TopK SAEs with a top n×k over a batch of n samples, so the number of
active latents varies per token at a fixed average. At inference the batch dependence is removed by a single global
threshold θ (the mean over batches of the smallest positive activation), applied as a JumpReLU. Evaluated on GPT-2 Small
(residual layer 8) and Gemma 2 2B (residual layer 12) with OpenWebText, widths 3k to 24k, k in {16, 32, 64}, Adam 3e-4,
1B tokens.

**Key result.** BatchTopK has lower normalised MSE and less CE degradation than TopK at every tested width and k on both
models. It beats JumpReLU on both measures on GPT-2 Small; on Gemma 2 2B it reconstructs better but has lower CE
degradation than JumpReLU only at k = 16. Active latents per token range from 1 (often BOS) to more than 80.

**Quote.** "we select the top n × k activations across the entire batch of n samples, setting all other activations to
zero." (sec. 3)

**Relevance to our paper.** The architecture of our SAE featurization and of Minder et al.'s crosscoder baseline; the
draft cites it correctly. One methodological point: our featurization must be fixed from M and applied per input to F,
which holds only with the paper's inference threshold θ (estimated on M), not with batch-level top-k on F's
activations, where F's larger or smaller activations would change which latents of other tokens fire. The draft does
not say which is used; it should state that θ from M is used. No bearing on the reuse question itself.

**For the digest.**
