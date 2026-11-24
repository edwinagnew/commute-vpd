---
slug: moskvoretskii2026-persona-vectors-pretraining
title: Tracing Persona Vectors Through LLM Pretraining
authors: Moskvoretskii et al.
year: 2026
id: 2605.13329
tags: [EH, SUPPORTS, DYNAMICS, CITE-FIX]
primary_tag: EH
in_our_bib: no
cited_in_draft: no
themes: [elicitation, training-dynamics]
read_from: "arXiv HTML v1: abstract, sections 2-6 (method, emergence, transfer through post-training, geometry, facets, extraction sensitivity)"
digest_buckets: [4-supports]
importance: 5
---

# Tracing Persona Vectors Through LLM Pretraining (Moskvoretskii 2026)

**What it does.** The paper extracts persona vectors (evil, sycophantic, impolite, humor; mean-difference directions after Chen et al.) at 17 pretraining checkpoints of OLMo-3-7B, with a replication on Apertus-8B. It steers each checkpoint with its own vector and, as the transfer test, steers the final base model and the SFT, DPO and RLVR instruct models with vectors taken from earlier base checkpoints. Steering strength is scaled to the local activation norm so that it is comparable across checkpoints.

**Key result.** Persona vectors become extractable within the first 0.22% of pretraining tokens, at a different onset for each trait. Vectors taken from very early base checkpoints still steer the fully post-trained OLMo-3-7B-Instruct, and the suppression of personas is concentrated at the DPO stage. The cosine of each checkpoint's vector to the final one rises from about 0.3, so the directions keep refining through pretraining, yet even early, distant vectors transfer.

**Quote.** "Pretraining sets the persona; post-training only tunes its volume, with suppression concentrated at DPO." (Sec. 4 takeaway)

**Relevance to our paper.** This is clean activation-level evidence for the elicitation hypothesis on a real SFT, DPO and RLVR pipeline. A base-model direction still controls the trait after post-training, and post-training changes its gain, which is the informal analogue of our "kept up to a positive scalar". The test is steering transfer, an intervention by value on activations. It does not show that the post-trained model computes the trait with the base model's mechanism, which is the gap our definition targets. It also does not touch our finding that trained fine-tunes turn writes: a base direction that still steers is compatible with rotated weights upstream. It belongs in the planned persona-selection paragraph as the best "activations say elicit" result, set against our mechanism-level reading.

**For the digest.**
- 4-supports: persona directions from base checkpoints steer the post-trained OLMo-3 models; post-training rescales ("tunes volume") rather than rebuilds them, by an activation-level reading.
