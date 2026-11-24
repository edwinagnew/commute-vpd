---
slug: lee2024-dpo-toxicity
title: "A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity"
authors: Andrew Lee et al.
year: 2024
id: 2401.01967
tags: [KNOWN-ANSWER, FT-INTERP, DIFFING, SUPPORTS, EH, ALIGNMENT, CONTRADICTS]
primary_tag: KNOWN-ANSWER
in_our_bib: lee2024dpo
cited_in_draft: yes
themes: [interpreting-finetunes, elicitation, reuse-definitions, model-diffing]
read_from: "arXiv HTML full (abstract, method, results, un-aligning section)"
digest_buckets: [2-known-answer-case, 4-supports, 4-contradicts]
importance: 7
---

# A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity (Lee 2024)

**What it does.** Asks how DPO lowers toxicity in GPT-2 medium. A linear toxicity probe on the last residual stream
picks out "toxic" MLP value vectors (the MLP neurons whose output weights align with the probe), and subtracting them
is shown to lower toxicity. GPT-2 medium is then DPO-trained on 24,576 toxic/non-toxic pairs (made with PPLM on
Wikitext-2 prompts), and the two models are compared at the levels of the weights, the neuron activations and the
residual stream.

**Key result.** Toxicity on RealToxicityPrompts falls from 0.453 to 0.208, and perplexity rises from 21.7 to 23.3. Every
weight matrix keeps a cosine similarity above 0.99 with its pre-DPO value, and the toxic neurons' key and value vectors
barely change. What changes is a distributed offset in the residual stream: it is spread over many value vectors in
earlier layers, and it moves inputs out of the region that activates the toxic neurons. Scaling the key vectors of 7
toxic neurons by 10 brings toxicity back to 0.458 (the base model reads 0.453), with perplexity unchanged.

**Quote.** "DPO does not remove the capability of generating toxic outputs, but learns an “offset”, distributed amongst
its layers, to “bypass” the regions that elicit toxicity." (Introduction)

**Relevance to our paper.** This is the closest realistic known-answer case in this batch, and the draft cites it only
for "DPO bypassing toxic vectors". The paper's own evidence comes from three sources: weights that barely move, the
activation shift, and an un-aligning intervention on 7 named neurons that restores base behaviour. Together they
support this reading: the fine-tuned behaviour runs through base neurons whose mechanism is kept, and what changes is
when those neurons fire. That is the situation our re-gating share is built to detect.

It is cheap to run in our setting. GPT-2 medium and their DPO recipe are public. On their pair, we could test whether
the re-gating share and ρ call the toxic neurons kept. The case also probes our Def. 1, in two ways:
- The behaviour-carrying change is many tiny additive shifts in other neurons' value vectors, not a rescaling. Our
  "kept up to a gain" clause might read those upstream neurons as changed, and could call the update mostly new on the
  needed set.
- The toxic neurons themselves are kept almost exactly.

Read the right way round, it bears on our "trained fine-tunes turn writes" finding. The paper reports the value-vector
shifts and the residual offset only as directions, not as rescalings.

**For the digest.**
- `2-known-answer-case`: GPT-2 medium + DPO (toxicity). Kept toxic neurons are bypassed, with independent evidence from weight cosines over 0.99 and a 7-neuron un-aligning that restores base toxicity. Runnable test of the re-gating share and ρ.
- `4-supports`: post-training re-gates existing components rather than removing them, which fits EH and the re-gating reading.
- `4-contradicts`: the change is distributed additive offsets, not gains. Our gain-only kept clause may misread this clear reuse case. Worth checking before claiming ρ "follows the mechanism".
