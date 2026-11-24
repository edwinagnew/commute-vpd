---
slug: qi2024-safety-few-tokens-deep
title: Safety Alignment Should Be Made More Than Just a Few Tokens Deep
authors: Xiangyu Qi et al.
year: 2024
id: 2406.05946
tags: [EH, SUPPORTS, ALIGNMENT, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation]
read_from: "arXiv HTML first ~100k chars (abstract, sections 2-4 incl. per-token KL, prefilling, data augmentation, constrained fine-tuning results)"
digest_buckets: [4-supports]
importance: 4
---

# Safety Alignment Should Be Made More Than Just a Few Tokens Deep (Qi 2024)

**What it does.** Compares aligned chat models with their base models: Llama-2-7B-Chat against Llama-2-7B, and Gemma-7B-IT
against Gemma-7B. It measures the per-token KL divergence between them on harmful responses, and it prefills refusal
prefixes on the base models. It then proposes two fixes: data augmentation that teaches the model to recover to a refusal
after a harmful prefix, and a token-wise constrained fine-tuning objective that limits drift on the first tokens. ICLR 2025
oral.

**Key result.** The divergence between the aligned and base models sits almost entirely in the first few response tokens.
Prefilling the base Llama-2 with "I cannot fulfill" drops its harmfulness from 68.6% to 5.4%, so the base model already
holds most of the refusal behaviour. Five prefilled harmful tokens raise the chat model's attack success rate to 42.1%
(2.8% after augmentation). On 100 harmful fine-tuning examples, standard SFT takes Llama-2 from 1.5% to 88.9%, and the
constrained objective holds it at 4.6%. The analysis is entirely output-level: no internal representations are examined.

**Quote.** Shallow alignment "adapts the base model's generative distribution only over the very first few output tokens" (Section 2)

**Relevance to our paper.** This is behavioural evidence for the selection view of post-training, from a real safety
fine-tune. It is the same kind of evidence as URIAL, which is in our draft, and it should sit beside URIAL in "Post-training
as selection". It also illustrates the gap our paper names: the evidence is behavioural and does not look inside the model.
It cannot say whether the refusal runs through a base mechanism. It is not a method we can compare against, and not a
known-answer case.

**For the digest.**
- `4-supports`: the aligned-vs-base difference is confined to the first few tokens, and the base model refuses when given a refusal prefix. This is behavioural support for EH and a candidate citation next to URIAL/LIMA (CITE-FIX: not in our bib).
