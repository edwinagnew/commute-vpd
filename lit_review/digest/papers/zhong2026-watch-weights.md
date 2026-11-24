---
slug: zhong2026-watch-weights
title: "Watch the Weights: Unsupervised Monitoring and Control of Fine-Tuned LLMs (WeightWatch)"
authors: Zhong et al. (Ziqian Zhong, Aditi Raghunathan)
year: 2026
id: 2508.00161
tags: [BASELINE, DIFFING, FT-INTERP, ALIGNMENT, CITE-FIX]
primary_tag: BASELINE
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines, interpreting-finetunes]
read_from: "arXiv HTML: abstract, intro, Sec. 2-5 (method, RLHF-poisoned case study, backdoor and unlearning results), related work, steering appendix lines"
digest_buckets: [1-missing-method]
importance: 6
---

# Watch the Weights: Unsupervised Monitoring and Control of Fine-Tuned LLMs (Zhong 2026)

**What it does.** A data-free diffing method. It takes the SVD of the weight difference ΔW = W_post - W_base of each
attention output projection and MLP down-projection, and keeps the top k = 20 left singular vectors per matrix as
"behavioral vectors". At inference it flags any token whose activation's cosine with a vector falls outside a running
range, and it can steer by projecting the activation off that vector. Tested on backdoored models (LoRA, full fine-tune,
poisoned RLHF), unlearned models (WHP, Zephyr-RMU, circuit breakers) and in-the-wild instruct models (OLMo 7B, Qwen 2.5
7B, Llama 3.1 8B).

**Key result.** It flags 93-100% of backdoor uses on first sight at under 1% false positives, and detects unlearned-topic
queries at 36-95%. Projecting off the vectors recovers 76.9% of Zephyr-RMU's original WMDP-Bio score, so the "unlearned"
knowledge is still there. In the instruct models the top directions point at maths, emoji use, Chinese ideological
content and Midjourney prompts, which they check against OLMo's public SFT data. Without the base model (SVD of W itself)
results range from perfect to near random.

**Quote.** "the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors." (Abstract)

**Relevance to our paper.** This is a weight-space diffing tool we neither compare against nor cite. The draft's claim
that its readings are new should engage it, and it is next to our effective-rank reading (both read the SVD of ΔW). It
runs on our setting at once: it needs only the two weight sets, so it costs nothing on the 67M model and on Qwen3-1.7B.
Its natural score for us: does the fine-tune's top singular direction align with a base direction already in use (reuse)
or with one the base hardly uses (new)? It would then face our known answers. Its own framing ("newly acquired
behaviors") is behavioural and does not separate a new mechanism from a reweighted one: the RMU result in fact shows
the fine-tune suppressing base knowledge along a low-rank direction, which reads as an edit on top of a kept mechanism.

**For the digest.**
- 1-missing-method: data-free SVD-of-ΔW diffing; runs on the 67M and Qwen as is; should be a baseline beside effective rank, and cited.
