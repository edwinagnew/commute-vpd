---
slug: lin2023-urial
title: "The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning"
authors: Lin et al.
year: 2023
id: 2312.01552
tags: [EH, DIFFING, BASELINE, SUPPORTS]
primary_tag: EH
in_our_bib: lin2024urial
cited_in_draft: yes
themes: [elicitation, model-diffing, baselines]
read_from: "arXiv HTML: abstract, intro, section 2 (token distribution shift), section 3 (URIAL), section 4.3 results"
digest_buckets: [1-missing-method, 4-supports]
importance: 5
---

# The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (Lin 2023)

**What it does.** Tests the Superficial Alignment Hypothesis by "token distribution shift": the aligned model
decodes greedily, the base model is run on the same prefix, and each position is classed by the base rank of the
aligned token (unshifted = 1, marginal = 2-3, shifted > 3), with KL and base probability per position. Pairs:
Llama-2-7B vs Llama-2-7B-chat (RLHF), vs Vicuna-7B (SFT), Mistral-7B vs Mistral-Instruct (SFT). It then proposes
URIAL, aligning a base model with a system prompt and three stylistic in-context examples, scored on 1,000 instructions
(just-eval-instruct) over six aspects by GPT-4.

**Key result.** On Llama-2-7B-chat, 77.7% of tokens are unshifted and 92.2% within the base model's top 3; shifted
ratios are 5-7% across the three pairs and concentrate on stylistic tokens ("However", "cannot", "Here", "Thank") and
safety disclaimers, and shrink later in the response. URIAL on Mistral-7B scores 4.63 against Mistral-7B-Instruct's
4.44, and on Llama-2-70B 4.74 against Llama-2-70B-chat's 4.67.

**Quote.** "alignment tuning primarily learns to adopt the language style of AI assistants, and that the knowledge
required for answering user queries predominantly comes from the base LLMs themselves" (Abstract)

**Relevance to our paper.** The draft cites it accurately for matching aligned models in context, but not for its
diffing measurement, which is the canonical behavioural base-versus-tuned comparison and the clearest example of the
"evidence so far is behavioural" the draft positions against. Token-shift share is essentially a per-position form of
our "base prior" baseline and could be computed on all twelve 67M fine-tunes and the Qwen runs for $0; we already find
the base model's loss on the training text tracks the raters as well as our cheap readings do, so this would sit with
that baseline. It cannot, by construction, separate two fine-tunes with the same outputs, which is exactly what our
definition is built to do; it is worth one sentence as the behavioural reading our definition refines.

**For the digest.**
- 1-missing-method: token-distribution shift (base rank / KL of the tuned model's tokens) is a cheap behavioural diff we could report beside the base-prior baseline.
- 4-supports: behavioural support for EH in chat tuning (most tokens unchanged, shift on style); bears on EH, not on our mechanistic claims.
