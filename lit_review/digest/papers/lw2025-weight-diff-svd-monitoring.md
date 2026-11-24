---
slug: lw2025-weight-diff-svd-monitoring
linkpost_of: zhong2026-watch-weights (2508.00161; cite the paper)
title: Weight-diff SVD for LLM Monitoring (LessWrong linkpost for "Watch the Weights", arXiv 2508.00161)
authors: Zhong & Raghunathan
year: 2025
id: https://www.lesswrong.com/posts/EPri6M2tMedbyahLq
tags: [BASELINE, DIFFING, ALIGNMENT]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: LessWrong post full (short linkpost); the paper's abstract and pp. 1-2 from the local PDF Zhong2026_watch_the_weights.pdf
digest_buckets: [1-missing-method]
importance: 2
---

# Weight-diff SVD for LLM Monitoring (Zhong 2025)

**What it does.** A 3-minute linkpost for the ICLR 2026 paper "Watch the Weights" (WeightWatch), which has its own
digest entry (`zhong2026-watch-weights`). It takes the difference of each layer's attention output and MLP
down-projection matrices between base and fine-tune, keeps the top singular vectors on the residual-stream side, and
monitors the cosine of live activations with them, flagging tokens outside a running range.

**Key result.** On simulated WildChat/UltraChat traffic it detects the first backdoor use with 44% to 100% accuracy at
~1% false positives (100% on first sight in 9 of 13 models), detects queries on unlearned content in 3 unlearned models
at 36% to 95%, and surfaces fine-tuning idiosyncrasies in OLMo (Midjourney prompts, marketing). The author calls it
"much more coarse-grained" than parameter decomposition.

**Quote.** "Top singular vectors on weight difference between base and fine-tuned models correspond to fine-tuned
behaviors." (TL;DR)

**Relevance to our paper.** It is the same work as the WeightWatch paper; engage with it through that entry and cite
the paper, not the post. The method is a data-free weight-level diff close to our effective-rank reading. It
could run on our 67M fine-tunes in minutes (SVD of ΔW, then the alignment of base subcomponents with the top singular
vectors), but it gives no reuse/new label, so it would be a feature for a baseline rather than a rival definition.

**For the digest.**
- 1-missing-method: weight-diff SVD (WeightWatch); duplicate of zhong2026-watch-weights, see that entry.
