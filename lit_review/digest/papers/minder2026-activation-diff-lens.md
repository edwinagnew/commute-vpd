---
slug: minder2026-activation-diff-lens
title: "Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences"
authors: Minder et al.
year: 2026
id: 2510.13900
tags: [BASELINE, DIFFING, ALIGNMENT]
primary_tag: BASELINE
in_our_bib: minder2025narrow
cited_in_draft: yes
themes: [model-diffing, baselines]
read_from: "arXiv HTML: abstract, intro, related work, Sec. 3 method (ADL: Patchscope/logit lens/steering on mean activation difference, interpretability agent), Sec. 4 results, mitigation summary"
digest_buckets: []
importance: 5
---

# Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences (Minder 2026)

**What it does.** It takes the mean difference of middle-layer residual activations between a narrowly fine-tuned model
and its base on the first 5 tokens of unrelated web text, then reads it with Patchscope and logit lens and steers the
fine-tuned model with it (the Activation Difference Lens, ADL). An LLM agent given these readings has to name the
fine-tuning objective. Organisms: synthetic-document false facts, emergent misalignment, subliminal learning and taboo
models, across Gemma, Llama and Qwen at 1B-32B.

**Key result.** 20-80% of the Patchscope tokens are relevant to the fine-tuning domain, and steering produces text like
the training data. The ADL agent names the broad objective for 91% of organisms, against 39% for a black-box agent with
50 interactions. The traces look like overfitting to a semantic concept shared by all samples, and mixing in pretraining
data mostly removes them.

**Quote.** "the common practice of using such models as a proxy for studying broader finetuning—such as chat-tuning—might not be realistic" (Abstract)

**Relevance to our paper.** The draft adapts it as the activation-difference score baseline, which is a fair use. Two
points for the write-up. First, ADL reads *what* a fine-tune is about, not whether it reused a mechanism, so its failure
on our known answers should be framed as "not built for this question", not as a refutation. Second, the paper's warning
applies to us: several of our 67M fine-tunes (French, Yoruba, lowercase) are narrow, single-concept fine-tunes of the
kind that leave a constant bias. A reading that does well on them may be picking up that bias. That is worth one
sentence where the draft reports realistic fine-tunes.

**For the digest.**
(no bucket: already a baseline. Note for the write-up: its narrow-fine-tune bias warning applies to our 67M fine-tunes.)
