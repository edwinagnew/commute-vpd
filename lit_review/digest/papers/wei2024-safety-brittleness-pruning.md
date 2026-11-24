---
slug: wei2024-safety-brittleness-pruning
title: "Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications"
authors: Wei et al.
year: 2024
id: 2402.05162
tags: [ALIGNMENT, BASELINE, FT-INTERP, EH]
primary_tag: ALIGNMENT
in_our_bib: no
cited_in_draft: no
themes: [interpreting-finetunes, baselines, other]
read_from: "arXiv abs page (abstract with numbers) + arXiv HTML: intro, section 2 (methods), sections 4.1-4.4, limitations, related work"
digest_buckets: [1-missing-method]
importance: 3
---

# Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications (Wei 2024)

**What it does.** Attributes safety behaviour (refusing AdvBench prompts) and utility (Alpaca-style instructions) to
weights of Llama2-7B-chat and 13B-chat, at the neuron level (SNIP and Wanda importance scores, isolated by set
difference) and at the rank level (ActSVD, a data-aware low-rank decomposition of each weight matrix, isolated by
orthogonal projection). It then removes the isolated safety regions, and tests whether freezing them protects against
fine-tuning attacks. It works within the chat model only; the base model is never compared.

**Key result.** The safety-only regions are sparse: about 3% of parameters and 2.5% of ranks. Removing them raises
attack success sharply while utility stays largely intact; attention-head probing localises worse than the weight
attribution. Freezing the safety-critical neurons during a fine-tuning attack resists only small attacks, which the
authors read (after Lee et al.) as the attack creating new pathways around the safety mechanism.

**Quote.** "This suggests that fine-tuning attacks may introduce new pathways that bypass the original safety
mechanisms in the model." (Section 1)

**Relevance to our paper.** Little direct bearing. It localises a post-trained behaviour within one model rather than
diffing base and fine-tune, and its "new pathways" reading of fine-tuning attacks is a hypothesis, not a measurement.
ActSVD's rank-one, activation-weighted split of each weight matrix is a cheap cousin of a parameter decomposition and
could be named as an alternative featurization in related work; the draft's claim that a decomposition "has not been
used to diff a fine-tune before" is unaffected, since Wei et al. do not diff. The alignment-applications note already
lists it under "safety is a thin wrapper", which overstates what it tests.

**For the digest.**
- 1-missing-method: ActSVD / SNIP-Wanda set-difference weight attribution localises a behaviour in weights; not a diff, could localise the fine-tune's behaviour in F's weights only.
