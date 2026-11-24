---
slug: ward2025-rank1-lora
title: Rank-1 LoRAs Encode Interpretable Reasoning Signals
authors: Ward et al.
year: 2025
id: 2511.06739
tags: [FT-INTERP, EH, SUPPORTS, BASELINE]
primary_tag: FT-INTERP
in_our_bib: no
cited_in_draft: no
themes: [elicitation, interpreting-finetunes]
read_from: arXiv HTML full text (workshop paper, sections 1-4 and appendices A-C)
digest_buckets: [4-supports, 1-missing-method]
importance: 4
---

# Rank-1 LoRAs Encode Interpretable Reasoning Signals (Ward 2025)

**What it does.** Trains a rank-1 LoRA on every MLP and attention projection of Qwen-2.5-32B-Instruct, on the s1k-1.1
set of 1,000 DeepSeek-R1 traces (5 epochs). Because each adapter is rank 1, each of its 448 components gives one scalar
activation per token. The authors treat these scalars as probes, autointerpret them with an LLM against base-model MLP
neurons, train a BatchTopK SAE on the 448-dimensional adapter state, and ablate components.

**Key result.** The rank-1 LoRA recovers 72.7% (AIME'24), 86.4% (MATH500) and 89.9% (GPQA-Diamond) of the gap between
the base model and a full fine-tune. Adapter activations are judged monosemantic about as often as MLP neurons, but
fire more on answers, math symbols and reasoning discourse. Among SAE features, 62% are judged "cleanly monosemantic".
Removing all MLP adapters hurts much more than removing all attention adapters; later layers matter most. Steering with
adapter directions was inconclusive: it needed more than 50 times the normal activation size.

**Quote.** "we show that the majority of new capabilities in reasoning models can be elicited by small, single-rank
changes to base model parameters" (Abstract)

**Relevance to our paper.** The paper gives behavioural support for the elicitation framing: a 32B reasoning gain
mostly fits into one direction per matrix. It bears on our effective-rank reading and on rank-one subcomponent views
of an update. It does not test reuse. What the LoRA reads and writes is labelled, but nothing checks whether those
directions are base-model mechanisms. A small update can still build a new mechanism, and our built "new" edits are
small too. Reading the adapter's own activations is a method we could cite as an alternative lens on an update (train a
rank-1 LoRA to the same data, then interpret it), but it answers "what changed", not "reused or new". Cite it at most
in the elicitation paragraph, with that caveat.

**For the digest.**
- 4-supports: a rank-1 LoRA recovers 73-90% of a full reasoning fine-tune on Qwen-2.5-32B, consistent with small re-weighting updates (behavioural only).
- 1-missing-method: "LoRA as a measurement device" (interpret rank-1 adapter activations); runnable on our 67M, but reads what changed, not reuse.
