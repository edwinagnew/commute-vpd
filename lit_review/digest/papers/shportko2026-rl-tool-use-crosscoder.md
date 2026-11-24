---
slug: shportko2026-rl-tool-use-crosscoder
title: "Localizing RL-Induced Tool Use to a Single Crosscoder Feature"
authors: Shportko et al.
year: 2026
id: 2606.26474
tags: [DIFFING, FT-INTERP, BASELINE]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, interpreting-finetunes, baselines]
read_from: "arXiv HTML: abstract, sec. 1-3 (DFC, sweep, scoring, steering), sec. 5-6 (spillover, filter vs sink, steering, geometry), limitations"
digest_buckets: [1-missing-method]
importance: 4
---

# Localizing RL-Induced Tool Use to a Single Crosscoder Feature (Shportko 2026)

**What it does.** Diffs Qwen2.5-3B against its ToolRL (RL for tool calling) fine-tune with Dedicated Feature Crosscoders
(DFC). A DFC is a crosscoder whose dictionary is split by gradient masks into RL-only, base-only and shared parts. It
sweeps 48 variants (36 DFC, 12 plain crosscoders) over dictionary size, top-k, exclusive share and an exclusive-latent
penalty. It scores tool-call format and correctness after passing activations through the crosscoder, and after
steering RL-only features (ranked by Cohen's d on tool vs general text) in the RL model's residual stream.

**Key result.** Reconstruction raises the RL model's tool correctness by 31.1 ± 9.7 points. Passing the base model's
activations through the crosscoder raises the base from 0% to 6.8% ("capability spillover"). Penalising the RL-only part
pushes tool signal into the shared part, so the authors call the partition "a filter", not "a sink". Steering one
RL-only feature (a tool-call template detector) gives +65 points, which a plain crosscoder reaches only with 33
features. Base-only steering gives 0, a harness check.

**Quote.** "We do not claim RL installs a tool-use ability the base model wholly lacks." (limitations)

**Relevance to our paper.** Another crosscoder reading of an RL fine-tune, here on a 3B Qwen close to our Qwen3-1.7B
scale. Its own evidence cuts both ways on reuse versus new, so it is not a known-answer case. The "RL-only" feature is a
template detector, and most of the spilled-over capability sits in the shared partition. That fits our point that
crosscoder exclusivity is not a reuse verdict. DFC is a crosscoder variant we do not run. Its masked exclusive partition
targets the bias toward shared latents that hurts small diffs, so it is a fair stronger-baseline candidate for our
planted control, where our BatchTopK crosscoder missed the plant up to about 100 times fine-tune size.

**For the digest.**
- 1-missing-method: Dedicated Feature Crosscoder (masked exclusive partitions); could run on our 67M/Qwen pairs as a stronger crosscoder baseline against the planted control; its own results show the exclusive part does not isolate the RL change.
