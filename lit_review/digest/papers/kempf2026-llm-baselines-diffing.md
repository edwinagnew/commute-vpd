---
slug: kempf2026-llm-baselines-diffing
title: "Simple LLM Baselines are Competitive for Model Diffing"
authors: Kempf et al.
year: 2026
id: 2602.10371
tags: [DIFFING, BASELINE, BACKGROUND]
primary_tag: DIFFING
in_our_bib: no
cited_in_draft: no
themes: [model-diffing, baselines]
read_from: arXiv HTML full main text (abstract, sections 1-5); appendices skimmed by heading only
digest_buckets: []
importance: 3
---

# Simple LLM Baselines are Competitive for Model Diffing (Kempf 2026)

**What it does.** Compares two API-only (black-box) model-diffing methods that turn output differences between two
models into natural-language hypotheses: an LLM pipeline (extract per-pair differences, embed, cluster, summarise,
after VibeCheck / Dunlap et al.) and the SAE pipeline of Jiang et al. (a "reader" LLM with an SAE reads both models'
responses; features with the largest activation-frequency gap are relabelled). It proposes held-out metrics
(frequency, accuracy, LLM-rated interestingness and abstraction, acceptance rate). Settings: Qwen-2.5-7B-Instruct
fine-tuned on risky financial advice (emergent misalignment), gemma-2-9b-it fine-tuned to assume a female user, and
two Gemini 2.5 Flash Lite revisions; 1,000 WildChat prompts, 500 held out.

**Key result.** The two methods are close on accuracy, frequency and interestingness; the LLM method gives more
abstract hypotheses and higher acceptance. Both find the misalignment model's harmful advice (100% accuracy, 3.4%
frequency) and its much shorter answers (68 vs 496 tokens). Neither finds the gender model's planted behaviour: it was
almost never verbalised on WildChat, so both report only side effects of the fine-tune.

**Quote.** "behavioral differences that do not manifest in model outputs cannot be surfaced regardless of method." (§4.1)

**Relevance to our paper.** This is behaviour-level diffing. It says what changed in outputs, not whether the change
runs through base mechanisms, so it does not compete with our readings of reuse. It is still useful in two ways. Its
"model organisms with known ground truth" are known in behaviour only, which fits our gap sentence that no diffing
method is scored on fine-tunes whose reuse status is known by construction. And the gender-model failure is a clean
example of what our introduction says: output-level evidence can miss what a fine-tune changed inside. It could not
run usefully on our 67M fine-tunes (the model's outputs are too weak to describe). At most a one-line citation in the
model-diffing related work, as the black-box branch.

**For the digest.** (none)
