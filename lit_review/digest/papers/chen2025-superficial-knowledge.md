---
slug: chen2025-superficial-knowledge
title: Extracting and Understanding the Superficial Knowledge in Alignment
authors: Chen et al.
year: 2025
id: 2502.04602
tags: [EH, BASELINE, DEFINITION, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, baselines, reuse-definitions]
read_from: "arXiv HTML: abstract, introduction, Section 2 (method, Table 1, 2.3 analysis), Appendix C table for Mistral"
digest_buckets: [1-missing-method, 3-definition, 4-supports]
importance: 5
---

# Extracting and Understanding the Superficial Knowledge in Alignment (Chen 2025)

**What it does.** Tests the Superficial Alignment Hypothesis quantitatively. "Superficial knowledge" is defined as what
can be gained by token restyling alone, without changing how the model relates tokens internally. They operationalise
it by keeping the base model's transformer frozen and distilling the aligned model into a residual ΔW on the final
linear projection (lm_head) only, with a KL loss to the aligned model's next-token distribution. Models: LLaMA2-7B/13B
vs their chat versions, plus Mistral-7B and Qwen; benchmarks GSM8K, Toxigen, AdvBench, TruthfulQA.

**Key result.** The head-only base model removes toxicity and harmful outputs entirely (7B Toxigen 0.77 to 0.00,
AdvBench harm rate 0.66 to 0.00) and recovers about 58% of the GSM gain and 78% of the TruthfulQA gain on average,
beating URIAL and LIMA. A gap remains on knowledge-heavy tasks (7B GSM: base 0.037, head-only 0.140, chat 0.230), and
the KL to the aligned model stays above zero at later response positions. The extracted head transfers to a larger
model and restores 88% of alignment effects after a de-aligning fine-tune.

**Quote.** "However, alignment is not entirely superficial." (Introduction)

**Relevance to our paper.** This is behavioural-plus-architecture evidence for a partial EH, of exactly the kind our
draft says does not look inside the model: it shows how much of the behaviour an output-layer-only change can
reproduce, not whether the fine-tune's internal mechanisms are the base model's. It should sit in our
"post-training as selection" related work next to URIAL and Raghavendra et al. Its head-only distillation is also a
cheap reading we could run as a baseline: the share of a fine-tune's gain (or KL) recovered by refitting only the
unembedding of the base model, a cousin of our gain-only (IA3-style) refits; it runs trivially on the 67M model and
our known-answer pairs (copy should read high, cipher low, if the baseline works). Its definition ("superficial" =
recoverable by restricting the change to the head) is a structural, location-based definition to contrast with our
mechanism-based one.

**For the digest.**
- 1-missing-method: head-only distillation share is a cheap "how superficial is this fine-tune" reading; could be scored on our built pairs.
- 3-definition: "superficial knowledge" defined by what a final-layer-only change recovers; contrast with our mechanism-kept definition.
- 4-supports: post-training gains are partly but not wholly re-styling of base computation; supports a graded rather than all-or-nothing EH.
