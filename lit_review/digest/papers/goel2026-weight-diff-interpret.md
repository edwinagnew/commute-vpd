---
slug: goel2026-weight-diff-interpret
title: Learning to Interpret Weight Differences in Language Models
authors: Goel et al.
year: 2026
id: 2510.05092
tags: [DIFFING, BASELINE, CITE-FIX]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: arXiv HTML (2510.05092v4; ICLR 2026 version) abstract, sections 1-4, 6-7; section 5 numbers from Table 6.1 and the old summary
digest_buckets: [1-missing-method]
importance: 4
---

# Learning to Interpret Weight Differences in Language Models (Goel 2026)

**What it does.** Diff Interpretation Tuning (DIT) trains a LoRA "DIT-adapter" on thousands of synthetic, labelled
fine-tunes, so that a fine-tuned model with the adapter applied answers a fixed question about how it was changed. The
two settings are hidden personas gated by a 3-digit trigger (4,760 rank-1 LoRAs) and knowledge of a news headline
(8,860 rank-8 LoRAs). The models are Qwen3 and Gemma3, 1B to 8B. The method is scored by an LLM judge against black-box
questioning baselines.

**Key result.** On 100 held-out hidden-topic diffs DIT scores 4.76/5 (Qwen3-4B) and 4.82/5 (Gemma3-4B), against about
1.2 for asking the model directly. This is comparable to a baseline given the trigger. It transfers from rank-1 to
higher-rank and partly to full fine-tunes. There is no transfer across behaviours: a topic adapter scores 1.25 on
news diffs (Table 6.1). Recovering the trigger fails completely (0 of 100).

**Quote.** "WeightDiffQA mitigates this issue because methods for solving it can be tested on synthetically crafted
triples (M, M′, q) where the answer to q is specified up-front" (Sec. 2)

**Relevance to our paper.** It is a weight-diff reader validated on fine-tunes whose answer is known by construction,
which is the same evaluation design as our built pairs. The answer it knows is what was learned, not whether it reuses
a base mechanism. So our claim "no diffing method scored on fine-tunes whose new or reused status is known by
construction" stands, but DIT is the nearest case and should be cited beside it so a reviewer does not raise it. It
cannot run as a baseline for us: the 67M Pile model cannot answer questions, and DIT does not generalise across
behaviours. Its related work also states the elicitation view ("finetuning often modulates existing capabilities"),
citing Jain and Prakash.

**For the digest.**
- 1-missing-method: introspective weight-diff reader with known-by-construction evaluation; not runnable on our 67M and answers "what", not "reused or new"; cite next to our gap claim.
