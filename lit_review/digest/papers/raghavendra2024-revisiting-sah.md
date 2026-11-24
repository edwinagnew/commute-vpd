---
slug: raghavendra2024-revisiting-sah
title: "Revisiting the Superficial Alignment Hypothesis"
authors: Raghavendra et al.
year: 2024
id: 2410.03717
tags: [EH, CONTRADICTS, DEFINITION]
primary_tag: EH
in_our_bib: raghavendra2024revisiting
cited_in_draft: yes
themes: [elicitation]
read_from: "arXiv HTML v1: abstract, sections 1-2 (claims and takeaways), section 3-5 results by grep (scaling, LIMA vs task, style vs reasoning errors, Facts100); appendices not read"
digest_buckets: [3-definition, 4-contradicts]
importance: 4
---

# Revisiting the Superficial Alignment Hypothesis (Raghavendra 2024)

**What it does.** Splits LIMA's Superficial Alignment Hypothesis into three claims: knowledge is learned entirely in
pre-training; post-training is mostly style and teaches no new capability; a few examples saturate a task. It tests
them by fine-tuning Llama-3, Llama-2 and Mistral base models of several sizes on growing task-specific sets (GSM8k,
SubQA multihop QA, coding, IFEval-style instruction following), with objective benchmarks instead of win rates. An
error analysis separates formatting errors from reasoning errors. A Facts100 set of post-cutoff news events tests
whether post-training helps a model take in new knowledge.

**Key result.** Task performance scales as a power law in the number of fine-tuning examples and does not saturate
after a handful. Style and formatting errors saturate within about 100 examples, while reasoning errors keep falling
and track accuracy. A 1k LIMA-style model wins on judge preference but scores 14.7% on GSM8k against 46.5% for a 1k
task-specific model. On Facts100, a reasoning-post-trained Llama-3-8B scores 32 (direct) and 27 (multihop) against the
base model's 8 and 12 before new knowledge is introduced, and integrates new facts better.

**Quote.** "Taken together, these results shed new light on the Superficial Alignment Hypothesis, suggesting that it
is, at best, an over-simplification." (Abstract)

**Relevance to our paper.** The draft cites it correctly (main.tex: SAH's style-only claim "that gains on reasoning
tasks have since challenged"). Its three-claim split is a useful foil for our definition, because EH as we state it
keeps their claim C1 (capability placed in pre-training) and drops C3 (style only). Saying so in one clause would make
the contrast with SAH sharper. Its evidence is purely behavioural: continued gains with more data are compatible with
either reuse (strengthening existing reasoning mechanisms) or new mechanisms. That is exactly the gap our paper claims
to fill, and the draft could say so explicitly when citing it. Nothing to fix.

**For the digest.**
- 3-definition: a clean three-claim decomposition of SAH (knowledge from pre-training / style only / few-shot saturation) to contrast with EH's mechanistic claim.
- 4-contradicts: contradicts the strong "style only" reading, not EH; behavioural evidence that post-training adds task ability with data.
